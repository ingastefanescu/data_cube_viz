function ydata = tsne_image(train_X, labels, no_dims, initial_dims, perplexity)
    if ~exist('labels', 'var')
        labels = [];
    end
    if ~exist('no_dims', 'var') || isempty(no_dims)
        no_dims = 2;
    end
    if ~exist('initial_dims', 'var') || isempty(initial_dims)
        initial_dims = min(50, size(train_X, 2));
    end
    if ~exist('perplexity', 'var') || isempty(perplexity)
        perplexity = 30;
    end
    
     % First check whether we already have an initial solution
    if numel(no_dims) > 1
        initial_solution = true;
        ydata = no_dims;
        no_dims = size(ydata, 2);
        perplexity = initial_dims;
    else
        initial_solution = false;
    end
    
    % Flatten the images
    flattenedImages = zeros(numel(msi_replaced), 2325 * 2086);
    for i = 1:numel(msi_replaced)
        flattenedImages(i, :) = reshape(msi_replaced{i}, 1, []);
    end
    
    % Perform preprocessing using PCA
    if ~initial_solution
        disp('Preprocessing data using PCA...');
        C = cov(flattenedImages);
        [M, lambda] = eig(C);
        [~, ind] = sort(diag(lambda), 'descend');
        M = M(:, ind(1:initial_dims));
        flattenedImages = bsxfun(@minus, flattenedImages, mean(flattenedImages, 1)) * M;
    end
    
    % Compute pairwise distance matrix
    sum_X = sum(flattenedImages .^ 2, 2);
    D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (flattenedImages * flattenedImages')));
    
    % Compute joint probabilities
    P = d2p(D, perplexity, 1e-5); % compute affinities using fixed perplexity
    
    % Run t-SNE
    if initial_solution
        ydata = tsne_p(P, labels, ydata);
    else
        ydata = tsne_p(P, labels, no_dims);
    end
end
   
