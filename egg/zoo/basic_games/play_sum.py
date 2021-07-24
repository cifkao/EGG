# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
import shlex

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import egg.core as core
from egg.core import Callback, Interaction, PrintValidationEvents
from egg.zoo.basic_games.architectures import SumReceiver, SumSender
from egg.zoo.basic_games.data_readers import AttrValClassDataset


# the following section specifies parameters that are specific to our games: we will also inherit the
# standard EGG parameters from https://github.com/facebookresearch/EGG/blob/master/egg/core/util.py
def get_params(params):
    parser = argparse.ArgumentParser()
    # arguments concerning the input data and how they are processed
    parser.add_argument(
        "--train_data", type=str, default=None, help="Path to the train data"
    )
    parser.add_argument(
        "--validation_data", type=str, default=None, help="Path to the validation data"
    )
    # (the following is only used in the reco game)
    parser.add_argument(
        "--n_attributes",
        type=int,
        default=None,
        help="Number of attributes in Sender input (must match data set, and it is only used in reco game)",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        default=None,
        help="Number of values for each attribute (must match data set)",
    )
    parser.add_argument(
        "--validation_batch_size",
        type=int,
        default=0,
        help="Batch size when processing validation data, whereas training data batch_size is controlled by batch_size (default: same as training data batch size)",
    )
    # arguments concerning the training method
    parser.add_argument(
        "--mode",
        type=str,
        default="rf",
        help="Selects whether Reinforce or Gumbel-Softmax relaxation is used for training {rf, gs} (default: rf)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="GS temperature for the sender, only relevant in Gumbel-Softmax (gs) mode (default: 1.0)",
    )
    parser.add_argument(
        "--sender_entropy_coeff",
        type=float,
        default=1e-1,
        help="Reinforce entropy regularization coefficient for Sender, only relevant in Reinforce (rf) mode (default: 1e-1)",
    )
    # arguments concerning the agent architectures
    parser.add_argument(
        "--sender_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_hidden",
        type=int,
        default=10,
        help="Size of the hidden layer of Receiver (default: 10)",
    )
    parser.add_argument(
        "--sender_layers",
        type=int,
        default=2,
        help="Number of hidden layers of Sender (default: 2)",
    )
    parser.add_argument(
        "--receiver_layers",
        type=int,
        default=2,
        help="Number of hidden layers of Receiver (default: 2)",
    )
    # arguments controlling the script output
    parser.add_argument(
        "--print_validation_events",
        default=False,
        action="store_true",
        help="If this flag is passed, at the end of training the script prints the input validation data, the corresponding messages produced by the Sender, and the output probabilities produced by the Receiver (default: do not print)",
    )
    args = core.init(parser, params)
    if args.validation_batch_size == 0:
        args.validation_batch_size = args.batch_size
    return args


def main(params, opts=None, train=False):
    if not opts:
        opts = get_params(params)
    print(opts, flush=True)

    if train and opts.checkpoint_dir:
        with open(Path(opts.checkpoint_dir) / "args", "w") as f:
            print(shlex.join(params), file=f)
        with open(Path(opts.checkpoint_dir) / "opts", "w") as f:
            print(repr(opts), file=f)

    # we are expecting 2 inputs by default
    n_attributes = (opts.n_attributes or 2)
    # number of features for the sender:
    n_features = n_attributes * opts.n_values
    # number of output classes = maximum summation result + 1:
    n_classes = n_attributes * (opts.n_values - 1) + 1

    def loss(
        sender_input, _message, _receiver_input, receiver_output, labels, _aux_input
    ):
        # in the case of the summation game, we have a standard classification task; we compute cross-entropy and accuracy.
        batch_size = sender_input.size(0)
        receiver_guesses = receiver_output.argmax(dim=1)
        acc = (receiver_guesses == labels).detach().float()
        loss = F.cross_entropy(receiver_output, labels, reduction="none")
        return loss, {"acc": acc}

    train_loader, test_loader = None, None
    if train:
        train_loader = DataLoader(
            AttrValClassDataset(
                path=opts.train_data,
                n_values=opts.n_values,
            ),
            batch_size=opts.batch_size,
            shuffle=True,
            num_workers=1,
        )
        test_loader = DataLoader(
            AttrValClassDataset(
                path=opts.validation_data,
                n_values=opts.n_values,
            ),
            batch_size=opts.validation_batch_size,
            shuffle=False,
            num_workers=1,
        )

    # now the sender and receiver agents. let's make them deeper since we don't have the RNN
    sender = SumSender(
        n_features=n_features,
        n_hidden=opts.sender_hidden,
        vocab_size=opts.vocab_size,
        n_layers=opts.sender_layers
    )
    receiver = SumReceiver(
        n_features=n_classes,
        n_hidden=opts.receiver_hidden,
        n_layers=opts.receiver_layers
    )

    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object

    # the Receiver wrapper takes the symbol produced by the Sender, embeds it and feeds it to the
    # core Receiver architecture we defined above (possibly with other Receiver input, as determined by the core architecture)
    # to generate the output.
    # here we use SymbolReceiverWrapper, which is compatible both with Gumbel-Softmax and Reinforce
    receiver = core.SymbolReceiverWrapper(
        receiver,
        vocab_size=opts.vocab_size,
        agent_input_size=opts.receiver_hidden
    )

    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
    if opts.mode.lower() == "gs":
        # in the following lines, we embed the Sender and Receiver architectures into standard EGG wrappers that are appropriate for Gumbel-Softmax optimization
        # the Sender wrapper takes the hidden layer produced by the core agent architecture we defined above when processing input, and samples a symbol
        # using Gumbel-Softmax
        sender = core.GumbelSoftmaxWrapper(
            sender,
            temperature=opts.temperature,
        )
        game = core.SymbolGameGS(sender, receiver, loss)
        # callback functions can be passed to the trainer object (see below) to operate at certain steps of training and validation
        # for example, the TemperatureUpdater (defined in callbacks.py in the core directory) will update the Gumbel-Softmax temperature hyperparameter
        # after each epoch
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:  # NB: any other string than gs will lead to rf training!
        # here, the interesting thing to note is that we use the same core architectures we defined above, but now we embed them in wrappers that are suited to
        # Reinforce-based optmization
        sender = core.ReinforceWrapper(sender)
        receiver = core.ReinforceDeterministicWrapper(receiver)
        game = core.SymbolGameReinforce(
            sender,
            receiver,
            loss,
            sender_entropy_coeff=opts.sender_entropy_coeff,
            receiver_entropy_coeff=0,
        )
        callbacks = []

    # we are almost ready to train: we define here an optimizer calling standard pytorch functionality
    optimizer = core.build_optimizer(game.parameters())
    # in the following statement, we finally instantiate the trainer object with all the components we defined (the game, the optimizer, the data
    # and the callbacks)
    if opts.print_validation_events == True:
        # we add a callback that will print loss and accuracy after each training and validation pass (see ConsoleLogger in callbacks.py in core directory)
        # if requested by the user, we will also print a detailed log of the validation pass after full training: look at PrintValidationEvents in
        # language_analysis.py (core directory)
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [
                core.ConsoleLogger(print_train_loss=True, as_json=True),
                core.PrintValidationEvents(n_epochs=opts.n_epochs),
            ],
        )
    else:
        trainer = core.Trainer(
            game=game,
            optimizer=optimizer,
            train_data=train_loader,
            validation_data=test_loader,
            callbacks=callbacks
            + [core.ConsoleLogger(print_train_loss=True, as_json=True)],
        )

    if train:
        # and finally we train!
        trainer.train(n_epochs=opts.n_epochs)

    return game


if __name__ == "__main__":
    import sys

    main(sys.argv[1:], train=True)
