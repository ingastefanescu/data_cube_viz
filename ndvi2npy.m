% Assuming your time series array is named 'timeSeriesImages'
% Replace this with the actual name of your array

% Specify the folder where you want to save the JPG files
outputFolder = 'ndvi';  % Change this to your desired folder path

% Check if the folder exists, and create it if necessary
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

% Get the dimensions of the time series array
[rows, columns, numberOfFrames] = size(ndvi_data);

% Iterate over each frame and save it as a separate JPG file
for frameIndex = 1:numberOfFrames
    % Extract the current frame
    currentFrame = ndvi_data(:, :, frameIndex);
    
    % Construct the file name for the current frame
    fileName = sprintf('image_%04d.jpg', frameIndex);  % You can customize the file name pattern
    
    % Construct the full file path
    filePath = fullfile(outputFolder, fileName);
    
    % Save the current frame as a JPG file
    imwrite(currentFrame, filePath, 'jpg');
end
