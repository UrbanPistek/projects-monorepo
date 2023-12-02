% utils script
clc
clear
close all

% load as struct
FileData = load('data/data_RANSAC.mat');

% export data from struct to csv
writematrix(FileData.data, 'data/data_RANSAC.csv');
