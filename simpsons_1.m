% Bristo Joemon 7/06/2022
% Create a neural network in MATLAB

close all
% Close the workspace
clear
% Clear the command wondow
clc

%%
% The data was uploaded into matlabdrive as online version matlab was used
% for this project
imdspath = fullfile(matlabdrive, 'simpsons01','img');

%%
%Data is read from the matlabdrive where each subfolder name is taken as
%the class name. As there are 5 folders each for one class with the name of
%the class as the folder name we use the below command to take the data in
%the each folder and give them the label name which is present as the
%foldername
imageds = imageDatastore(imdspath, 'IncludeSubfolders', true,'LabelSource','foldernames')

%%
% The below command shows the sample images
figure;
perm = randperm(100,20); for i = 1:10
subplot(2,5,i); imshow(imageds.Files{perm(i)});
end

%%
% To get an understanding about the data and label count
labelCount = countEachLabel(imageds)

%%
%To check the image size as it is required for building the neural network
img = readimage(imageds,10); size(img)

%%
% Spliting the data into training and testing
numTrainFiles = 60;
[imdsTrain,imdsValidation] = splitEachLabel(imageds,numTrainFiles,'randomize');

%%
layers = [
imageInputLayer([200 200 3])
convolution2dLayer(3,8,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
%leakyReluLayer
%reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,16,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
%leakyReluLayer
%reluLayer
maxPooling2dLayer(2,'Stride',2)
convolution2dLayer(3,32,'Padding','same')
batchNormalizationLayer
%tanhLayer
swishLayer
%leakyReluLayer
%reluLayer
fullyConnectedLayer(5)
softmaxLayer
classificationLayer];
%%
% validation layer data is augmented into a single format so that it can be
% fed into the network. It is difficult to feed the network with varying
% pixel sizes to the netwrork. 
imdsValidation2 = augmentedImageDatastore([200 200],imdsValidation)

%%
options = trainingOptions('sgdm', ...
 'InitialLearnRate',0.01, ...
 'MaxEpochs',100, ...
 'Shuffle','every-epoch', ...
 'ValidationData',imdsValidation2, ...
 'ValidationFrequency',10, ...
 'Verbose',false, ...
 'Plots','training-progress');

%%
% Augmentinng the images with random differences to avoid overfitting while
% training the data as the dataset is small
imageAugmenter = imageDataAugmenter('RandXReflection',1, ...
    'RandYReflection',1,...
    ...'FillValue',[200 200 3],...
    ...'RandXShear',[-90,90],...
    ...'RandYShear',[-90,90],...
    ...'RandScale',[0.6 1.2],...
    'RandRotation',[-20,20], ...
    'RandXTranslation',[-11 11], ...
    'RandYTranslation',[-11 11])

%%
% Augmenting the training data
auds = augmentedImageDatastore([200 200],imageds,'DataAugmentation',imageAugmenter);
%%
% training the network
net = trainNetwork(auds,layers,options);

%%
% Checking the accuracy
YPred = classify(net,imdsValidation2);
YValidation = imdsValidation.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);

%%
display(accuracy)

%%
% confusion matrix is plotted to understand how the model worked and where
% the model made wrong classifications
confusionchart(YValidation,YPred)
