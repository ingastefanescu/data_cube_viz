% Script that will create time series from one pixel (both MSI and NDVI)
clear all;

%% Crop images

ndvi = Tiff('NDVI_SITS_GIS.tif','r');
ndvi_data = read(ndvi);

% Dimensions of the NDVI image series
[numRows, numCols, numImages] = size(ndvi_data);

% Define crop coordinates
x1 = 500;  % x-coordinate of top-left corner
y1 = 1950; % y-coordinate of top-left corner
x2 = 600;  % x-coordinate of bottom-right corner
y2 = 2100; % y-coordinate of bottom-right corner

% Compute cropped dimensions
cropHeight = y2 - y1;
cropWidth = x2 - x1;

% Initialize matrix for cropped NDVI images
cropped_ndvi_data = single(zeros(cropHeight, cropWidth, numImages));

% Perform cropping on each NDVI image
for k = 1:numImages
    % Extract the k-th NDVI image from the series
    img = ndvi_data(:,:,k);
    
    % Perform cropping
    cropped_img = img(y1:y1+cropHeight-1, x1:x1+cropWidth-1);
    
    % Store the cropped image in the output array
    cropped_ndvi_data(:,:,k) = cropped_img;
end

% Display one of the original and cropped NDVI images for verification
figure;
subplot(1, 2, 1);
imshow(ndvi_data(:,:,1), []);
title('Original NDVI Image');

subplot(1, 2, 2);
imshow(cropped_ndvi_data(:,:,1), []);
title('Cropped NDVI Image');

%% Create time series
x = 3;
y = 19;

ndvi_replaced = normalize_image_series(cropped_ndvi_data, 0, 1);

% Initialize an array to store the pixel values for each sample
ndvi_timeSeries = zeros(114, 1);

% Iterate over each sample
for i = 1:114
    % Extract a single pixel from each image
    % For example, let's extract the pixel at position (100, 100) for each image
    pixel_value = ndvi_replaced(x,y,i);
    
    % Store the pixel value in the time series array
    ndvi_timeSeries(i) = pixel_value;
end

% Plot the time series
figure(1)
plot(1:114, ndvi_timeSeries);
xlabel('Sample');
ylabel('Pixel Value');
title('NDVI Time Series');