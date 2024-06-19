function croppedTimeSeries = cropImageEdgesTimeSeries(images, cropAmount)
    % Determine the size of the time series
    [rows, cols, numFrames] = size(images);

    % Initialize the croppedTimeSeries array
    croppedTimeSeries = zeros(rows - 2*cropAmount, cols - 2*cropAmount, numFrames);

    % Crop each image in the time series
    for i = 1:numFrames
        % Get the current image
        currentImage = images(:,:,i);

        % Crop the image
        croppedImage = currentImage(cropAmount+1 : rows-cropAmount, cropAmount+1 : cols-cropAmount);

        % Store the cropped image in the result array
        croppedTimeSeries(:,:,i) = croppedImage;
    end
end
