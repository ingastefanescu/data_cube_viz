clear;

% Load an example image (replace 'example.jpg' with your image file path)
image = imread('D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\ndvi\image_0114.jpg');

% Display the image
figure;
imshow(image);
title('Click on the image to get coordinates');

% Get handle to the current figure
fig = gcf;

% Set up the mouse click callback
set(fig, 'WindowButtonDownFcn', @imageClickCallback);

% Mouse click callback function
function imageClickCallback(~, ~)
    % Get the coordinates of the click relative to the image
    coordinates = get(gca, 'CurrentPoint');
    
    % Extract X and Y coordinates (considering potential inversion)
    x = round(coordinates(1, 1));  % X-coordinate
    y = round(coordinates(1, 2));  % Y-coordinate
    
    % Display the corrected coordinates in the command window
    fprintf('Clicked Point - X: %d, Y: %d\n', x, y);
    
    % Mark the clicked point on the image
    hold on;
    plot(x, y, 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end
