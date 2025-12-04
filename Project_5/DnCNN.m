%% DnCNN Image Denoising Script (User selects noisy image)
clc; clear; close all;

%% 1. Ask user to select the noisy image
[filename, filepath] = uigetfile({'*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff', ...
    'Image Files (*.png, *.jpg, *.jpeg, *.bmp, *.tif, *.tiff)'}, ...
    'Select the noisy image');

if isequal(filename, 0)
    disp('User canceled the selection.');
    return;
end

noisyPath = fullfile(filepath, filename);
fprintf('Selected file: %s\n', noisyPath);

%% 2. Load noisy image
noisyImg = im2double(imread(noisyPath));

% If the image is RGB, DnCNN can handle it, but if you want grayscale only:
% noisyImg = rgb2gray(noisyImg);

%% 3. Load DnCNN pretrained network
% MATLAB will auto-prompt to download if missing
net = denoisingNetwork('dncnn');

%% 4. Denoise using DnCNN
denoisedImg = denoiseImage(noisyImg, net);

%% 5. Display results
figure;
subplot(1,2,1);
imshow(noisyImg, []);
title('Noisy Image');

subplot(1,2,2);
imshow(denoisedImg, []);
title('DnCNN Denoised Image');

%% 6. Save output next to original
outputName = [filepath, 'denoised_', filename];
imwrite(denoisedImg, outputName);

fprintf('Denoised image saved as: %s\n', outputName);
