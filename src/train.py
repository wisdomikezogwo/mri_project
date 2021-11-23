import os
import random

import tqdm
import pandas as pd
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset

import argparse
from model import get_model
from dataset.brats_dataset import get_datasets
from dataset.config import get_train_val_df

parser = argparse.ArgumentParser(description='Brats Training')

parser.add_argument('-n_train', '--n_train_patients', default=50, type=int, metavar='N',
                    help='number of patients to use for Training from TrainingSet')

parser.add_argument('-n_val', '--n_valid_patients', default=10, type=int, metavar='N',
                    help='number of patients to use for Testing the data (actual testing)')

parser.add_argument('-in', '--in_channels', default=4, type=int, metavar='N')

parser.add_argument('-out', '--out_channels', default=4, type=int, metavar='N')

parser.add_argument('--init_feats', default=8, help='base number of features for Unet (x2 per downsampling)', type=int)

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--fold', default=0, type=int, help="Split number")

parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 1)')

parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--seed', default=577, help="seed for train/val split")

parser.add_argument('--data_dir', default='', type=str, metavar='PATH',
                    help='path to dataset')


def main(args):
    # Set random seed for reproduciablity
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    num_epochs = args.epochs

    model = get_model(args.in_channels, args.out_channels, init_features=args.init_feats)
    criterion = torch.nn.CrossEntropyLoss()

    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=args.fold, shuffle=True, random_state=args.seed)

    resize_shape = (144, 144, 144)  # hardcoded
    train_df, valid_df = get_train_val_df(args.n_train, n_val=args.n_val)
    transformed_dataset_train, transformed_dataset_valid = get_datasets(args.in_channels,
                                                                        resize_shape, train_df, valid_df)

    dataset = ConcatDataset([transformed_dataset_train, transformed_dataset_valid])

    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        dataloader_train = DataLoader(dataset, batch_size=args.b, sampler=train_subsampler, num_workers=0)
        dataloader_valid = DataLoader(dataset, batch_size=args.b, sampler=test_subsampler, num_workers=0)

        # Init the neural network
        network = model
        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)

        # Run the training loop for defined number of epochs
        for epoch in range(0, num_epochs):

            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            if dataloader_train is None or optimizer is None:
                print('None')
                break  # NotImplementedError
            for i, data in enumerate(tqdm.tqdm(dataloader_train)):
                # Get inputs
                (image, seg_image), ID = data
                inputs = torch.squeeze(torch.permute(image, (0, 4, 1, 2, 3)))  #
                label = torch.squeeze(torch.permute(seg_image, (0, 4, 1, 2, 3)))  # , 0)
                inputs, label = inputs.cuda(), label.cuda()  # add this line
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = criterion(outputs, label.long())
                print('Loss:', loss.item())
                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 500 == 499:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 500))
                    current_loss = 0.0
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Saving the model
        save_path = f'./model-fold-{fold}.pth'
        torch.save(network.state_dict(), 'drive/MyDrive/Colab Notebooks/')

        # Print about testing
        print('Starting testing')
        # Evaluationfor this fold
        correct, total = 0, 0
        with torch.no_grad():

            # Iterate over the test data and generate predictions
            for i, data in enumerate(dataloader_valid, 0):
                # Get inputs
                inputs, targets = data
                inputs = torch.squeeze(torch.permute(inputs, (0, 4, 1, 2, 3)))  #
                label = torch.squeeze(torch.permute(targets, (0, 4, 1, 2, 3)))  # , 0)
                inputs, label = inputs.cuda(), label.cuda()  # add this line

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct


if __name__ == '__main__':
    arguments = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arguments.devices
    main(arguments)
