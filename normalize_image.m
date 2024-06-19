function img_normalized = normalize_image(img, new_min_val, new_max_val)
    % Convert the image to double precision
    img_double = double(img);
    
    % Compute the minimum and maximum pixel values in the image
    min_val = min(img_double(:));
    max_val = max(img_double(:));
    
    % Perform dynamic range normalization to scale pixel values to [0, 1]
    img_normalized = (img_double - min_val) / (max_val - min_val);
    
    % Optionally scale pixel values to a custom target range [new_min_val, new_max_val]
    if nargin > 2  % Check if custom range is specified
        img_normalized = img_normalized * (new_max_val - new_min_val) + new_min_val;
    end
end