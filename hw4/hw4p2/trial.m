function value = trial(i,ver_padded_pixels,hor_padded_pixels,zero_svMap)


zero_padded_svMap = zero_svMap(1:100,1:100)
pre_neighbors_i = [];
for x = 2:50
    for y = 2 : 50
        x
        y
        %zero_padded_svMap(x,y)
        if zero_padded_svMap(x,y) == i
            submatrix_i = zero_padded_svMap(x-1:x+1,y-1:y+1)
            pre_neighbors_i = [pre_neighbors_i; unique(submatrix_i)]
        end
    end
end
neighbors_i = unique(pre_neighbors_i)
    if neighbors_i(1) == 0
        trimmed_neighbors_i = neighbors_i(2:end);
    end
end