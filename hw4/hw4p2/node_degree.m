function avg_degree = node_degree(adj_matrix)

%%% We can obtain the degree of a vertex from an adjacency matrix by adding
%%% the entries in the corresponding column of the vertex in the matrix. We
%%% will add the entries in every column and from these numbers we can get
%%% the average node degree.

size_matrix = size(adj_matrix);
no_columns = size_matrix(2);

% This part of the code adds the entries in each column and adds the value
% of each column.
degree_sum = 0;
for i = 1:no_columns
    column = adj_matrix(:,i);
    pre_sum = sum(column);
    degree_sum = degree_sum + pre_sum;
end

% Finding the degree. 

avg_degree = degree_sum/no_columns;

end