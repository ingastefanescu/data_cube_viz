clc, clear all;

% Load an example image (replace 'example.jpg' with your image file)
img = imread('D:\OneDrive - Universitatea Politehnica Bucuresti\Andrei\Facultate\Master\DISSERTATION\work\msi\image_0001.jpg');

% Display the original image
figure;
imshow(img);
title('Original Image');

% Define the coordinates of the specific pixel to highlight
row = 2100;  % Row index of the pixel
col = 800;  % Column index of the pixel

% Highlight the specific pixel with a marker (red circle)
hold on;
plot(col, row, 'ro', 'MarkerSize', 10, 'LineWidth', 2);  % Plot red circle at pixel location
hold off;

% Annotate the highlighted pixel with its coordinates
text(col + 10, row + 10, sprintf('(%d, %d)', row, col), 'Color', 'red', 'FontSize', 12, 'FontWeight', 'bold');

% Optionally, draw a rectangle around the specific pixel
rectangle('Position', [col-5, row-5, 10, 10], 'EdgeColor', 'red', 'LineWidth', 2);

% Add legend or annotation to the image
legend('Highlighted Pixel', 'Location', 'SouthEast');  % Add legend for the highlighted pixel
