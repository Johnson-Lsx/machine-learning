function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
%思路：我们要将被录取的学生和未被录取的学生在图上表示出来
%被录取的学生采用'+'表示，未被录取的采用'o'表示
%分析一个具体的数据点，某个学生两门成绩为（34，78），被录取了（y = 1）
%那么他在图像里就是一个横坐标为34，纵坐标为78的点，用'+'标记
%进一步推广，为了画出所有被录取的学生，我们需要找到所有y = 1的数据，再找到对应的X，将横纵坐标确定
pos = find(y == 1);
neg = find(y == 0);
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','y','MarkerSize',7);
% =========================================================================



hold off;

end
