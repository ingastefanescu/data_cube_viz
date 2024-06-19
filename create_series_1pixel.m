% Script that will create time series from one pixel (both MSI and NDVI)
clc, clear all;

%% Pixel coordinates

% 2100 and 800 for initial one (gr een)

x = 2028;
y = 854;
%% MSI

msi = Tiff('MSI_SITS_GIS.tif','r');
msi_data = read(msi);
%figure(1), imshow(msi_data(:,:,114));

replacement_value = 0;
msi_replaced = remove_nan(msi_data, replacement_value);
msi_replaced = normalize_image_series(msi_replaced, 0, 1);

% Initialize an array to store the pixel values for each sample
msi_timeSeries = zeros(114, 1);

% Iterate over each sample
for i = 1:114
    % Extract a single pixel from each image
    % For example, let's extract the pixel at position (100, 100) for each image
    pixel_value = msi_replaced(x,y,i);
    
    % Store the pixel value in the time series array
    msi_timeSeries(i) = pixel_value;
end

% Plot the time series
figure(1)
plot(1:114, msi_timeSeries); 
xlabel('Sample');
ylabel('Pixel Value');
title('MSI Time Series');

%% NDVI

ndvi = Tiff('NDVI_SITS_GIS.tif','r');

ndvi_data = read(ndvi);
%figure(2), imshow(ndvi_data(:,:,114));

replacement_value = 0;
ndvi_replaced = remove_nan(ndvi_data, replacement_value);
ndvi_replaced = normalize_image_series(ndvi_replaced, 0, 1);

% Initialize an array to store the pixel values for each sample
ndvi_timeSeries = zeros(114, 1);

% Iterate over each sample
for i = 1:114
    % Extract a single pixel from each image
    % For example, let's extract the pixel at position (100, 100) for each image
    pixel_value = ndvi_replaced(x, y, i);
    
    % Store the pixel value in the time series array
    ndvi_timeSeries(i) = pixel_value;
end

% Plot the time series
figure(2)
plot(1:114, ndvi_timeSeries);
xlabel('Sample');
ylabel('Pixel Value');
title('NDVI Time Series');
