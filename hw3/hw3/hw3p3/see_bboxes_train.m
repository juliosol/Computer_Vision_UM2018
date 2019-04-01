load('/home/julio/Documents/Michigan/CS/Computer_vision/HW/hw3/hw3/hw3p3/train_SVHN/digitStruct.mat')
for i = 1:length(digitStruct)
    im = imread([digitStruct(i).name]);
    for j = 1:length(digitStruct(i).bbox)
        [height, width] = size(im);
        aa = max(digitStruct(i).bbox(j).top+1,1);
        bb = min(digitStruct(i).bbox(j).top+digitStruct(i).bbox(j).height, height);
        cc = max(digitStruct(i).bbox(j).left+1,1);
        dd = min(digitStruct(i).bbox(j).left+digitStruct(i).bbox(j).width, width);
        
        imshow(im(aa:bb, cc:dd, :));
        fprintf('%d\n',digitStruct(i).bbox(j).label );
        pause;
    end
    
end