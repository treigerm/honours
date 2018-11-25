function stain_normalization(sourceimage, targetimage, outfilename)
verbose = 1;  

[SourceImage,map] = imread(sourceimage);
if ~isempty(map)
    SourceImage = ind2rgb(SourceImage,map);
end

[TargetImage,map] = imread(targetimage);
if ~isempty(map)
    TargetImage = ind2rgb(X,map);
end

[ NormSM ] = Norm(SourceImage, TargetImage, 'SCD', [], verbose);
imwrite(NormSM, outfilename)
end