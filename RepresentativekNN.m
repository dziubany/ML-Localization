% Script to apply repknn algorithm on real data of sound-based localization
% input: 
% 272 set of 10000 sound measurements at a sample rate of 10 kHz (1 s)
% - 80 training sets (5 for each label from 1 to 16, datasets 1-80)
% - 192 test sets for prediction (12 for each area, datasets 81-272)
% - 272 coordinates, 1 for each data set, sorted by time of measurement
% output:
% - plot of label-depending colored training datasets (squares) and 
%   predicted test datasets (circles) with number of label
% parameter:
% - useToolbox: 1 to use Matlab Toolbox 0 for COSY kNN & kMeans
% - representatives: set number of representatives (1<representatives<5)
% - neighbors: set number of neighbors (>0)   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear workspace
clear all;
useToolbox = 0;
representatives = 3;
neighbors = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GET DATA
% load coordinates
Coord = load('koords_paper.csv');
% path to files with datasets - please check
path = 'dataPaper\';
% get all fileproperties in directory
files = dir(path);
% get filenames
filenames = {files.name};
% count files
amountFiles = (numel(files)-2);
% declare help variables length raw data, used data
lengthRawData = 10000;
lengthData = 2000;
% declare data containing matrices
RawData = zeros(lengthRawData, amountFiles);
Data = zeros(lengthData, amountFiles);
% move raw data from files to raw data matrix
% for each file/column
for k = 1 : amountFiles
    % create path from prefix and filename
    fullpath = strcat(path,filenames{k+2});
    % load raw data 
    RawData(:,k) = load(fullpath);
end
% sensor sends data with offset, get mean from each raw dataset to adjust 
means = mean(RawData);
% adjust each raw dataset by mean value
for i = 1:amountFiles
    RawData(:,i) = RawData(:,i) - means(:,i);
end
% some datasets containing noise at beginning, clear these parts
RawData([1:400],5) = 0;
RawData([1:400],26) = 0;
RawData([1:400],34) = 0;
RawData([1:400],63) = 0;
RawData([1:400],182) = 0;
RawData([1:400],224) = 0;
% filter timeframe of 2000 samples (200 ms) from raw data for each file
for j = 1:amountFiles
    % check values for threshold
    for i = 1:lengthRawData
        % if absolut value bigger 50, save index 
        if (abs(RawData(i,j)) > 50)
            z = i - 1;
            break;
        end
    end
    % move raw data from threshold
    for h = 1:lengthData
        Data(h,j) = RawData((z+h-3),j);
    end
    % normalize data
    Data(:,j) = Data(:,j)/norm(Data(:,j));
    z = 0;
end
% END GET DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SET TRAINING DATA
% set size training data
train = 80;
% number of training clusters
parts = 16;
% units per cluster
trainunits = train/parts;
% create labelarray, 1-16 for each training area, 0 for testsets
label = zeros(1,amountFiles);
for i = 1 : parts
    for j = 1 : trainunits
        label(((i-1)*trainunits)+j) = i;
    end
end
% create trainingdatasets
M1 = [];
M2 = [];
M3 = [];
M4 = [];
M5 = [];
M6 = [];
M7 = [];
M8 = [];
M9 = [];
M10 = [];
M11 = [];
M12 = [];
M13 = [];
M14 = [];
M15 = [];
M16 = [];
% sort training data depending on label 
for i = 1:train
    if (label(i)==1)
        M1 = [M1, Data(:,i)];
    elseif (label(i)==2)
        M2 = [M2, Data(:,i)];
    elseif (label(i)==3)
        M3 = [M3, Data(:,i)];
    elseif (label(i)==4)
        M4 = [M4, Data(:,i)];
    elseif (label(i)==5)
        M5 = [M5, Data(:,i)];
    elseif (label(i)==6)
        M6 = [M6, Data(:,i)];
    elseif (label(i)==7)
        M7 = [M7, Data(:,i)];
    elseif (label(i)==8)
        M8 = [M8, Data(:,i)];
    elseif (label(i)==9)
        M9 = [M9, Data(:,i)];
    elseif (label(i)==10)
        M10 = [M10, Data(:,i)];
    elseif (label(i)==11)
        M11 = [M11, Data(:,i)];
    elseif (label(i)==12)
        M12 = [M12, Data(:,i)];
    elseif (label(i)==13)
        M13 = [M13, Data(:,i)];
    elseif (label(i)==14)
        M14 = [M14, Data(:,i)];
    elseif (label(i)==15)
        M15 = [M15, Data(:,i)];
    elseif (label(i)==16)
        M16 = [M16, Data(:,i)];
    end
end
% END SET TRAINING DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GET REPRESENTATIVES FROM TRAINING DATA
% clustering to get representatives for each label
if useToolbox == 1
    [idx1,R1] = kmeans(M1',representatives,'Distance','cosine');
    [idx2,R2] = kmeans(M2',representatives,'Distance','cosine');
    [idx3,R3] = kmeans(M3',representatives,'Distance','cosine');
    [idx4,R4] = kmeans(M4',representatives,'Distance','cosine');
    [idx5,R5] = kmeans(M5',representatives,'Distance','cosine');
    [idx6,R6] = kmeans(M6',representatives,'Distance','cosine');
    [idx7,R7] = kmeans(M7',representatives,'Distance','cosine');
    [idx8,R8] = kmeans(M8',representatives,'Distance','cosine');
    [idx9,R9] = kmeans(M9',representatives,'Distance','cosine');
    [idx10,R10] = kmeans(M10',representatives,'Distance','cosine');
    [idx11,R11] = kmeans(M11',representatives,'Distance','cosine');
    [idx12,R12] = kmeans(M12',representatives,'Distance','cosine');
    [idx13,R13] = kmeans(M13',representatives,'Distance','cosine');
    [idx14,R14] = kmeans(M14',representatives,'Distance','cosine');
    [idx15,R15] = kmeans(M15',representatives,'Distance','cosine');
    [idx16,R16] = kmeans(M16',representatives,'Distance','cosine');
else
    % clustering to get representatives for each label
    [R1] = kMeansCOSY(M1', representatives, 1);
    [R2] = kMeansCOSY(M2', representatives, 1);
    [R3] = kMeansCOSY(M3', representatives, 1);
    [R4] = kMeansCOSY(M4', representatives, 1);
    [R5] = kMeansCOSY(M5', representatives, 1);
    [R6] = kMeansCOSY(M6', representatives, 1);
    [R7] = kMeansCOSY(M7', representatives, 1);
    [R8] = kMeansCOSY(M8', representatives, 1);
    [R9] = kMeansCOSY(M9', representatives, 1);
    [R10] = kMeansCOSY(M10', representatives, 1);
    [R11] = kMeansCOSY(M11', representatives, 1);
    [R12] = kMeansCOSY(M12', representatives, 1);
    [R13] = kMeansCOSY(M13', representatives, 1);
    [R14] = kMeansCOSY(M14', representatives, 1);
    [R15] = kMeansCOSY(M15', representatives, 1);
    [R16] = kMeansCOSY(M16', representatives, 1);
end
% Repräsentanten normieren
for j = 1 : representatives
    R1(j,:) = R1(j,:)/norm(R1(j,:));
    R2(j,:) = R2(j,:)/norm(R2(j,:));
    R3(j,:) = R3(j,:)/norm(R3(j,:));
    R4(j,:) = R4(j,:)/norm(R4(j,:));
    R5(j,:) = R5(j,:)/norm(R5(j,:));
    R6(j,:) = R6(j,:)/norm(R6(j,:));
    R7(j,:) = R7(j,:)/norm(R7(j,:));
    R8(j,:) = R8(j,:)/norm(R8(j,:));
    R9(j,:) = R9(j,:)/norm(R9(j,:));
    R10(j,:) = R10(j,:)/norm(R10(j,:));
    R11(j,:) = R11(j,:)/norm(R11(j,:));
    R12(j,:) = R12(j,:)/norm(R12(j,:));
    R13(j,:) = R13(j,:)/norm(R13(j,:));
    R14(j,:) = R14(j,:)/norm(R14(j,:));
    R15(j,:) = R15(j,:)/norm(R15(j,:));
    R16(j,:) = R16(j,:)/norm(R16(j,:));
end


% END GET REPRESENTATIVES FROM TRAINING DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREDICT TEST DATA
% set label for representatives
sizelabel2 = representatives*parts;
label2 = ones(1,sizelabel2);
for i = 1 : parts
    for j = 1 : representatives
        label2(((i-1)*representatives)+j) = i; % 6-10
    end
end
if useToolbox == 1
    % apply k-nearest neighbors with Toolbox
    mdl = fitcknn([R1;R2;R3;R4;R5;R6;R7;R8;R9;R10;R11;R12;R13;R14;R15;R16],label2,'Distance','cosine');
    % set number auf neighbors
    mdl.NumNeighbors = neighbors;
    mdl.Distance = 'cosine';
    mdl.BreakTies = 'nearest';
    dist = mdl.Distance;
else
    distance = 1;
    if distance == 1
        dist = 'cosine';
    elseif distance == 2
        dist = 'euclidean';
    elseif distance == 3
        dist = 'sqeuclidean';
    end
end

% END PREDICT TEST DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

quote = 0;
tested = 0;
correct = 0;
s = zeros(1, length(Data(:,1)));
for y = 1:amountFiles
    if (label(y) == 0)
        tested = tested + 1;
        if useToolbox == 1
            % set s to the label of the next representatives to actual dataset  
            s(y) = predict(mdl,Data(:,y)');
        else
            R = [R1;R2;R3;R4;R5;R6;R7;R8;R9;R10;R11;R12;R13;R14;R15;R16];
            s(y) = kNNCOSY(R,label2,neighbors,Data(:,y),1);
        end
        areal = fix((y-81)/12) + 1;
        if(s(y) == areal)
            correct = correct + 1;
        end
    end
end
quote=correct/tested;

% PLOT RESULTS
figure('Name',dist);
if useToolbox == 1
    title(sprintf('Representative kNN with Toolbox\nCorrect: %.2f', quote*100));
else
    title(sprintf('Representative kNN without Toolbox\nCorrect: %.2f', quote*100));
end

hold
% set size and position of figure
set(gcf, 'Position', [0, 0, 1000, 1000])
xlabel('[cm]');
ylabel('[cm]'); 
% plot grid
set(gca,'xtick',[0:15:60])
set(gca,'ytick',[0:15:60])
grid on 
% offsets to adjust numbers to circles and squares
xoffset = 0.52;
yoffset = 0.1;
% check all files
for y = 1:amountFiles
        % if no training data
    if (label(y) == 0) 
        % plot color and number at position of dataset depending on s
        if (s(y)==1)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'r');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'13')
        elseif (s(y)==2)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'b');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'9')
        elseif (s(y)==3)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'g');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'5')
        elseif (s(y)==4)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'y');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'1')
        elseif (s(y)==5)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'm');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'14')
        elseif (s(y)==6)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'c');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'10')
        elseif (s(y)==7)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'k');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'6')
        elseif (s(y)==8)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', [1 0.4 0.6]);
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'2')
        elseif (s(y)==9)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'r');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'15')
        elseif (s(y)==10)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'b');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'11')
        elseif (s(y)==11)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'g');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'7')
        elseif (s(y)==12)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'y');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'3')
        elseif (s(y)==13)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'm');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'16')
        elseif (s(y)==14)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'c');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'12')
        elseif (s(y)==15)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'k');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'8')
        elseif (s(y)==16)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', [1 0.4 0.6]);
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'4')
        end
    % if training data
    % plot color and number at position of dataset depending on label
    elseif (label(y) == 1)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','r');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'13')
    elseif(label(y) == 2)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','b');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'9')
    elseif(label(y) == 3)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','g');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'5')
    elseif(label(y) == 4)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','y');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'1')
    elseif(label(y) == 5)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','m');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'14')
    elseif(label(y) == 6)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','c');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'10')
    elseif(label(y) == 7)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','k');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'6')
    elseif(label(y) == 8)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color',[1 0.4 0.6]);
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'2')
    elseif(label(y) == 9)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','r');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'15')
    elseif(label(y) == 10)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','b');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'11')
    elseif(label(y) == 11)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','g');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'7')
    elseif(label(y) == 12)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','y');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'3')
    elseif(label(y) == 13)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','m');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'16')
    elseif(label(y) == 14)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','c');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'12')
    elseif(label(y) == 15)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color','k');
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'8')
    elseif(label(y) == 16)
        plot(Coord(y,1),Coord(y,2),'s', 'LineWidth', 2, 'MarkerSize', 20,'Color',[1 0.4 0.6]);
        text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'4')
    end
end
% END PLOT RESULTS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
