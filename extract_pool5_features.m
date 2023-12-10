clear

imagedata='oxford5k_image';
net=vgg16;
netname='vgg16';
layer='pool5';
extract_net_features(imagedata,net,netname,layer)




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function []=extract_net_features(imagedata,net,netname,layer)

imagePath = dir(fullfile('G:\zhangbojian\matlab_paper\data\VGG16\datasets\',imagedata,'*.jpg'));
minsize=224;
folderPath = fullfile('..\data\',netname,'\',layer,'\',strtok(imagedata, '_'));

if exist(folderPath, 'dir')~=7
    mkdir(folderPath);
    disp('Folder created successfully!');
else
    disp('Folder already exists.');
end

parfor i=1:length(imagePath)
    imgPath = [imagePath(i).folder,'\',imagePath(i).name];
    im = imread(imgPath);
    [h,w,~]=size(im);
    
    if w<minsize || h<minsize
        im = imresize(im, minsize/min(h,w));
    end   
    pool5 = activations(net,im,layer,'OutputAs','channels');
    parsave([folderPath,'\',erase(imagePath(i).name,'.jpg'),'.mat'],pool5);
    if mod(i,1000) == 0
        i
    end
end
end
