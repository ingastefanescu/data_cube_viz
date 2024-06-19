%% MSI
clc, clear all;

msi = Tiff('MSI_SITS_GIS.tif','r');
msi_data = read(msi);
%figure(1), imshow(msi_data(:,:,114));

% image = msi_data(:,:,1);
% % Specify the amount of pixels to be cropped from the edges
% cropAmount = 35; % Adjust this value as needed
% 
% % Determine the size of the image
% [rows, cols] = size(image);
% 
% % Define the region of interest
% roi = image(cropAmount+1 : rows-cropAmount, cropAmount+1 : cols-cropAmount);
% 
% % Display the cropped image
% imshow(roi);


% cropAmount = 50
% msi_cropped = croppedTimeSeries(msi_data, cropAmount

replacement_value = 0;
msi_replaced = remove_nan(msi_data, replacement_value);

msi_mean = mean(mean(msi_replaced, 1 ), 2);
msi_avg = reshape(msi_mean, 1, []);
msi_tran = msi_avg';
msi_final = (msi_tran - min(msi_tran)) / (max(msi_tran) - min(msi_tran));
% msi_nan = isnan(msi_data) | isinf(msi_data);
% data(msi_nan) = 0;

%% NDVI
% clear, all clc;

ndvi = Tiff('NDVI_SITS_GIS.tif','r');

ndvi_data = read(ndvi);
%figure(2), imshow(ndvi_data(:,:,114));

replacement_value = 0;
ndvi_replaced = remove_nan(ndvi_data, replacement_value);

ndvi_mean = mean(mean(ndvi_replaced, 1 ), 2);
ndvi_avg = reshape(ndvi_mean, 1, []);
ndvi_final = ndvi_avg';

