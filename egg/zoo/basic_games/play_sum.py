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
    parser.add_argument(
        "--n_attributes",
        type=int,
        default=None,
        help="Number of attributes (operands in the sum game) in Sender input (default: 2)",
    )
    parser.add_argument(
        "--n_values",
        type=int,
        default=None,
        help="Number of values for each attribute (operand)",
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
        default=1,
        help="Number of hidden layers of Sender (default: 2)",
    )
    parser.add_argument(
        "--receiver_layers",
        type=int,
        default=1,
        help="Number of hidden layers of Receiver (default: 2)",
    )
    parser.add_argument(
        "--rnn",
        action="store_true",
        help="Use RNN game instead of single-symbol game",
    )
    parser.add_argument(
        "--sender_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Sender {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--receiver_cell",
        type=str,
        default="rnn",
        help="Type of the cell used for Receiver {rnn, gru, lstm} (default: rnn)",
    )
    parser.add_argument(
        "--sender_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds symbols produced at previous step in Sender (default: 10)",
    )
    parser.add_argument(
        "--receiver_embedding",
        type=int,
        default=10,
        help="Output dimensionality of the layer that embeds the message symbols for Receiver (default: 10)",
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


# allowing to pass either parsed or unparsed parameters, and whether we want to train or only create the game.
# this is so that we can load the game in a Jupyter notebook by calling main()
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

    # now the sender and receiver agents. we have the option to make them deeper if we don't use the RNN
    sender = SumSender(
        n_features=n_features,
        n_hidden=opts.sender_hidden,
        # if we are going to use the RNN wrapper, we output sender_hidden units, otherwise we
        # directly project to vocab_size dimensions for symbol prediction
        n_output=opts.sender_hidden if opts.rnn else opts.vocab_size,
        n_layers=opts.sender_layers
    )
    receiver = SumReceiver(
        n_features=n_classes,
        n_hidden=opts.receiver_hidden,
        n_layers=opts.receiver_layers
    )

    # now, we instantiate the full sender and receiver architectures, and connect them and the loss into a game object

    # the Receiver wrapper takes the symbol produced by the Sender, embeds it and feeds it to the
    # core Receiver architecture we defined above to generate the output.
    if not opts.rnn:
        # here we use SymbolReceiverWrapper, which is compatible both with Gumbel-Softmax and Reinforce
        receiver = core.SymbolReceiverWrapper(
            receiver,
            vocab_size=opts.vocab_size,
            agent_input_size=opts.receiver_hidden
        )

    # the implementation differs slightly depending on whether communication is optimized via Gumbel-Softmax ('gs') or Reinforce ('rf', default)
    if opts.mode.lower() == "gs":
        if opts.rnn:
            raise NotImplementedError()
        sender = core.GumbelSoftmaxWrapper(
            sender,
            temperature=opts.temperature,
        )
        game = core.SymbolGameGS(sender, receiver, loss)
        callbacks = [core.TemperatureUpdater(agent=sender, decay=0.9, minimum=0.1)]
    else:
        if opts.rnn:  # multi-symbol game, use RNN
            sender = core.RnnSenderReinforce(
                sender,
                vocab_size=opts.vocab_size,
                embed_dim=opts.sender_embedding,
                hidden_size=opts.sender_hidden,
                cell=opts.sender_cell,
                max_len=opts.max_len,
            )
            receiver = core.RnnReceiverDeterministic(
                receiver,
                vocab_size=opts.vocab_size,
                embed_dim=opts.receiver_embedding,
                hidden_size=opts.receiver_hidden,
                cell=opts.receiver_cell,
            )
            game = core.SenderReceiverRnnReinforce(
                sender,
                receiver,
                loss,
                sender_entropy_coeff=opts.sender_entropy_coeff,
                receiver_entropy_coeff=0,
            )
        else:  # single-symbol game
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
