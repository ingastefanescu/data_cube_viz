function processedImages = remove_nan(images, replaceValue)
    % Replace Inf and NaN values with a specific value
    processedImages = images;
    processedImages(isinf(processedImages) | isnan(processedImages)) = replaceValue;
end