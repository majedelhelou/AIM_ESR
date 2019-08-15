%======================================================================
% Extreme image completion demo
% Copyright (C) 2016 Ecole Polytechnique Federale de Lausanne
% File created by Radhakrishna Achanta
% This file is meant to demonstrate the extreme imgage completion
% algorithm called Filtering by Adaptive Normalization (FAN) presented
% the paper:
%
% "Extreme Image Completion", Radhakrishna Achanta, Nikolaos
% Arvanitopoulos, Sabine Susstrunk. ICASSP 2017, New Orleans, USA.
% 
%======================================================================

close all; clearvars; clc;
for ID = 0:9
    imgHR = imread( ['trainHR/000' num2str(ID) '.png']);

    % Create the LR version, bicubic down x16
    imgLR = imresize(imgHR,1/16);

    imgLR_YCbCr = rgb2ycbcr(imgLR);


    % Interpolate x16 with bicubic interp. & anti-aliasing
    imgHR_bic = imresize(imgLR,16);
    imgHR_lanc = imresize(imgLR,16,'lanczos3');

    imgHR_bic_YCbCr = imresize(imgLR_YCbCr,16);
    imgHR_bic_YCbCr = ycbcr2rgb(imgHR_bic_YCbCr);
    imgHR_lanc_YCbCr = imresize(imgLR_YCbCr,16,'lanczos3');
    imgHR_lanc_YCbCr = ycbcr2rgb(imgHR_lanc_YCbCr);

    % Create fused images:
    imgHR_bic_avg = 0.5 * imgHR_bic + 0.5 * imgHR_bic_YCbCr;
    imgHR_lanc_avg = 0.5 * imgHR_lanc + 0.5 * imgHR_lanc_YCbCr;
    imgHR_4_avg = 0.25 * imgHR_bic + 0.25 * imgHR_bic_YCbCr + 0.25 * imgHR_lanc + 0.25 * imgHR_lanc_YCbCr;
    
    % Compute MSE
    MSE_bic = mean(mean(mean((imgHR - imgHR_bic).^2)));
    MSE_lanc = mean(mean(mean((imgHR - imgHR_lanc).^2)));
    MSE_bic_YCbCr = mean(mean(mean((imgHR - imgHR_bic_YCbCr).^2)));
    MSE_lanc_YCbCr = mean(mean(mean((imgHR - imgHR_lanc_YCbCr).^2)));

    MSE_bic_avg = mean(mean(mean((imgHR - imgHR_bic_avg).^2)));
    MSE_lanc_avg = mean(mean(mean((imgHR - imgHR_lanc_avg).^2)));
    MSE_4_avg = mean(mean(mean((imgHR - imgHR_4_avg).^2)));
    
    disp(['MSE_bic, MSE_lanc, MSE_bic_YCbCr, MSE_lanc_YCbCr']);
    disp([MSE_bic, MSE_lanc, MSE_bic_YCbCr, MSE_lanc_YCbCr]);
    disp(['MSE of average bic RGB+YCBCr : ' num2str(MSE_bic_avg)]);
    disp(['MSE of average lanc RGB+YCBCr : ' num2str(MSE_lanc_avg)]);
    disp(['MSE of 4 average bic+lanc & RGB+YCBCr : ' num2str(MSE_4_avg)]);
    disp('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');
%     figure;
%     imshow(imgHR);
end


%%
for ID = 1501:1501
    disp(ID)
    imgLR = imread( ['validationLR/' num2str(ID) '.png']);
    imgLR_YCbCr = rgb2ycbcr(imgLR);


    % Interpolate x16 with bicubic interp. & anti-aliasing
    imgHR_lanc = imresize(imgLR,16,'lanczos3');

    imgHR_lanc_YCbCr = imresize(imgLR_YCbCr,16,'lanczos3');
    imgHR_lanc_YCbCr = ycbcr2rgb(imgHR_lanc_YCbCr);

    % Create fused images:
    imgHR_lanc_avg = 0.5 * imgHR_lanc + 0.5 * imgHR_lanc_YCbCr;
    
    % Crop 1000 and save:
    sz = size(imgHR_lanc_avg);
    Ic = imcrop(imgHR_lanc_avg, [sz(2)/2-499 sz(1)/2-499 999 999]);    
    imwrite(Ic, ['validationHRResultCropped/' num2str(ID) '.png']);      
%     figure;
%     imshow(imgHR);
end




%% Interpolate with fan
[ht wd dt] = size(imgHR_bic);
imgHR_fan = 0 * imgHR_bic;
imgHR_fan(8:16:end,8:16:end,:) = imgLR;

mask = uint8(zeros(ht, wd));
mask(8:16:end,8:16:end,:) = 1;
imgHR_fan = fan_func(imgHR_fan, mask);

%% Plot
figure;
subplot(221), imshow(imgHR);
subplot(222), imshow(imgLR);
subplot(223), imshow(imgHR_bic);
subplot(224), imshow(imgHR_fan);

MSE_bic = mean(mean((imgHR - imgHR_bic).^2));
MSE_fan = mean(mean((imgHR - imgHR_fan).^2));

%%
percentage = 0.01;%choose percentage of retained original pixels

s = size(img);
sz = s(1)*s(2);
rng('default');%choose seed for random number generator
randvec = randperm(sz,sz);
numpixels = round(sz*percentage);
randind = randvec(1:numpixels);
M = zeros([s(1) s(2)]);%mask
M(randind) = 1;
%--------------------------------------------------------------------------
tic
outimg = fan_func(img,uint8(M));% FAN (EFAN is about twice as fast)
toc
%--------------------------------------------------------------------------
figure;
subplot(1,3,1), imshow(img);
subplot(1,3,2), imshow(M);
subplot(1,3,3), imshow(outimg);

