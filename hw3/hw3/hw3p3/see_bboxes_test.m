load digitStruct_test.mat
for i = 1:length(digitStruct_test)
    im = imread([digitStruct_test(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        aa = max(digitStruct_test(i).bbox(j).top+1,1);
        bb = min(digitStruct_test(i).bbox(j).top+digitStruct_test(i).bbox(j).height, height);
        cc = max(digitStruct_test(i).bbox(j).left+1,1);
        dd = min(digitStruct_test(i).bbox(j).left+digitStruct_test(i).bbox(j).width, width);
        
        imshow(im(aa:bb, cc:dd, :));
        fprintf('%d\n',digitStruct_test(i).bbox(j).label );
        pause;
    end
end
