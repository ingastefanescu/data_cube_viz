function img_series_normalized = normalize_image_series(img_series, new_min_val, new_max_val)
    % Input:
    %   img_series: 3D array of images (height x width x num_images)
    %   new_min_val (optional): Minimum value of the target normalization range
    %   new_max_val (optional): Maximum value of the target normalization range
    % Output:
    %   img_series_normalized: Normalized 3D array of images

    % Get the number of images in the series
    num_images = size(img_series, 3);

    % Initialize an empty array to store normalized images
    img_series_normalized = zeros(size(img_series));

    % Loop through each image in the series and normalize
    for i = 1:num_images
        % Extract the current image from the series
        img = img_series(:, :, i);
        
        % Normalize the current image using the normalize_image function
        img_normalized = normalize_image(img, new_min_val, new_max_val);
        
        % Store the normalized image back into the series
        img_series_normalized(:, :, i) = img_normalized;
    end
end
