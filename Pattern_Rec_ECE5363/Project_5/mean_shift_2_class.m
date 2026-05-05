%% Mean Shift Demonstrated Like a Gradient Plot on 2 Clusters
% This script shows mean shift as motion uphill on a density surface.
% It uses two clusters so you can see paths converge to different modes.

clear; clc; close all;

%% -----------------------------
% 1. Generate two 2D clusters
% -----------------------------
rng(7);

n1 = 100;
n2 = 100;

C1 = [randn(n1,1)*0.7 - 2.5, randn(n1,1)*0.6 + 1.0];
C2 = [randn(n2,1)*0.8 + 2.5, randn(n2,1)*0.7 - 1.2];

X = [C1; C2];

%% -----------------------------
% 2. Mean shift parameters
% -----------------------------
bandwidth = 1.3;
tol = 1e-3;
maxIter = 40;
mergeThresh = bandwidth / 2;

%% -----------------------------
% 3. Create grid for density estimation
% -----------------------------
xMin = min(X(:,1)) - 2;
xMax = max(X(:,1)) + 2;
yMin = min(X(:,2)) - 2;
yMax = max(X(:,2)) + 2;

gridSize = 100;

[xGrid, yGrid] = meshgrid( ...
    linspace(xMin, xMax, gridSize), ...
    linspace(yMin, yMax, gridSize));

density = zeros(size(xGrid));

%% -----------------------------
% 4. Estimate density with Gaussian kernels
% -----------------------------
for i = 1:size(X,1)
    dx = xGrid - X(i,1);
    dy = yGrid - X(i,2);
    distSq = dx.^2 + dy.^2;

    density = density + exp(-distSq / (2 * bandwidth^2));
end

density = density / size(X,1);

%% -----------------------------
% 5. Compute density gradient
% -----------------------------
[gradY, gradX] = gradient(density, ...
    yGrid(2,1) - yGrid(1,1), ...
    xGrid(1,2) - xGrid(1,1));

gradMag = sqrt(gradX.^2 + gradY.^2);
gradX_norm = gradX ./ (gradMag + eps);
gradY_norm = gradY ./ (gradMag + eps);

%% -----------------------------
% 6. Run mean shift for every point
% -----------------------------
n = size(X,1);
shiftedPoints = zeros(size(X));
allPaths = cell(n,1);

for i = 1:n
    currentPoint = X(i,:);
    path = currentPoint;

    for iter = 1:maxIter
        distances = sqrt(sum((X - currentPoint).^2, 2));
        inBandwidth = distances <= bandwidth;

        newPoint = mean(X(inBandwidth, :), 1);
        path = [path; newPoint];

        if norm(newPoint - currentPoint) < tol
            break;
        end

        currentPoint = newPoint;
    end

    shiftedPoints(i,:) = currentPoint;
    allPaths{i} = path;
end

%% -----------------------------
% 7. Merge converged points into modes/clusters
% -----------------------------
clusterCenters = [];
labels = zeros(n,1);

for i = 1:n
    pointMode = shiftedPoints(i,:);

    if isempty(clusterCenters)
        clusterCenters = pointMode;
        labels(i) = 1;
    else
        distancesToCenters = sqrt(sum((clusterCenters - pointMode).^2, 2));
        [minDist, nearestCluster] = min(distancesToCenters);

        if minDist < mergeThresh
            labels(i) = nearestCluster;
            clusterCenters(nearestCluster,:) = ...
                mean([clusterCenters(nearestCluster,:); pointMode], 1);
        else
            clusterCenters = [clusterCenters; pointMode];
            labels(i) = size(clusterCenters,1);
        end
    end
end

numClusters = size(clusterCenters,1);

%% -----------------------------
% 8. Pick example paths from both clusters
% -----------------------------
% Choose points far from each cluster center so path movement is visible.
dist1 = sqrt(sum((C1 - mean(C1,1)).^2, 2));
dist2 = sqrt(sum((C2 - mean(C2,1)).^2, 2));

[~, idx1] = sort(dist1, 'descend');
[~, idx2] = sort(dist2, 'descend');

% Two example points from each cluster
examplePoints = [idx1(1); idx1(2); n1 + idx2(1); n1 + idx2(2)];

%% -----------------------------
% 9. Plot density contour + gradient field + mean shift paths
% -----------------------------
figure;
hold on;

% Density contour
contourf(xGrid, yGrid, density, 25, 'LineColor', 'none');
colorbar;

% Original data points
scatter(X(:,1), X(:,2), 28, 'k', 'filled', ...
    'MarkerFaceAlpha', 0.45);

% Gradient arrows
step = 5;
quiver( ...
    xGrid(1:step:end, 1:step:end), ...
    yGrid(1:step:end, 1:step:end), ...
    gradX_norm(1:step:end, 1:step:end), ...
    gradY_norm(1:step:end, 1:step:end), ...
    0.35, 'w', 'LineWidth', 1.0);

% Mean shift paths
colors = lines(length(examplePoints));

for k = 1:length(examplePoints)
    path = allPaths{examplePoints(k)};

    plot(path(:,1), path(:,2), '-o', ...
        'Color', colors(k,:), ...
        'MarkerFaceColor', colors(k,:), ...
        'LineWidth', 2.5, ...
        'MarkerSize', 6);

    % Start point
    scatter(path(1,1), path(1,2), ...
        140, colors(k,:), ...
        'filled', 'MarkerEdgeColor', 'k');

    % End point
    scatter(path(end,1), path(end,2), ...
        170, colors(k,:), ...
        'd', 'filled', 'MarkerEdgeColor', 'k');
end

% Plot density modes
scatter(clusterCenters(:,1), clusterCenters(:,2), ...
    260, 'kx', 'LineWidth', 3);

title(sprintf('Mean Shift as Gradient Ascent on a 2-Cluster Density Surface, Bandwidth = %.2f', bandwidth));
xlabel('Feature 1');
ylabel('Feature 2');
grid on;
axis equal;
hold off;

%% -----------------------------
% 10. Plot resulting clusters formed
% -----------------------------
figure;
hold on;

scatter(X(:,1), X(:,2), 55, labels, 'filled');
scatter(clusterCenters(:,1), clusterCenters(:,2), ...
    250, 'kx', 'LineWidth', 3);

title(sprintf('Resulting Clusters Formed by Mean Shift (%d Clusters)', numClusters));
xlabel('Feature 1');
ylabel('Feature 2');
grid on;
axis equal;
colorbar;
hold off;

%% -----------------------------
% 11. Optional 3D density surface plot
% -----------------------------
figure;
hold on;

surf(xGrid, yGrid, density, ...
    'EdgeColor', 'none', ...
    'FaceAlpha', 0.9);

scatter3(X(:,1), X(:,2), zeros(size(X,1),1), ...
    20, 'k', 'filled');

for k = 1:length(examplePoints)
    path = allPaths{examplePoints(k)};
    pathDensity = interp2(xGrid, yGrid, density, path(:,1), path(:,2));

    plot3(path(:,1), path(:,2), pathDensity, '-o', ...
        'Color', colors(k,:), ...
        'MarkerFaceColor', colors(k,:), ...
        'LineWidth', 2.2, ...
        'MarkerSize', 5);
end

scatter3(clusterCenters(:,1), clusterCenters(:,2), ...
    interp2(xGrid, yGrid, density, clusterCenters(:,1), clusterCenters(:,2)), ...
    260, 'kx', 'LineWidth', 3);

title('Mean Shift Paths Moving Uphill on the 2-Cluster KDE Surface');
xlabel('Feature 1');
ylabel('Feature 2');
zlabel('Estimated Density');
grid on;
view(45, 35);
hold off;

%% -----------------------------
% 12. Print summary
% -----------------------------
fprintf('Bandwidth: %.3f\n', bandwidth);
fprintf('Number of clusters found: %d\n', numClusters);
disp('Cluster centers:');
disp(clusterCenters);

for k = 1:length(examplePoints)
    path = allPaths{examplePoints(k)};
    fprintf('Path %d: start [%.3f, %.3f] -> end [%.3f, %.3f], class = %d, iterations = %d\n', ...
        k, path(1,1), path(1,2), path(end,1), path(end,2), ...
        labels(examplePoints(k)), size(path,1)-1);
end