function ydata = tsne_images(images, labels, no_dims, resizeFactor, perplexity)
    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('resizeFactor', 'var') || isempty(resizeFactor)
        resizeFactor = 0.1;  % Adjust the resize factor as needed
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    
    % Resize the images
    resizedImages = cell(size(images));

    for i = 1:numel(images)
        resizedImages{i} = imresize(images{i}, resizeFactor);
    end

    % Convert the resized images to a 4D array
    imageArray = cat(4, resizedImages{:});

    % Reshape the 4D array to 2D for t-SNE
    reshapedImages = reshape(imageArray, [], size(imageArray, 4));

    % Perform t-SNE
    ydata = tsne(reshapedImages', labels, no_dims, perplexity);
end