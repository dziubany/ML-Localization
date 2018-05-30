% Script to apply repknn algorithm on real data of sound-based localization
% input: 
% 272 sound measurements with length 1000 at a sample rate of 10 kHz (1 s)
% - 80 training samples (5 for each label from 1 to 16, datasets 1-80)
% - 192 test samples for prediction (12 for each area, datasets 81-272)
% - 272 coordinates, 1 for each data set, sorted by time of measurement
% output:
% - plot of label-depending colored training data (squares) and 
%   predicted test data (circles) with number of label
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% clear workspace
clear all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GET DATA
% path to file with coordinates - please check
path = strcat(userpath, '\ML-Localization-master\koords_paper.csv');
% read coordinates
Coord = load(path);
% path to files with datasets - please check
path = strcat(userpath, '\ML-Localization-master\dataPaper\');
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
% some measurements containing noise at beginning, clear these parts
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
% set number representatives
representatives = 3;
% clustering to get representatives for each label
[idx1,R1] = kmeans(M1',representatives,'Distance','cosine','Start','sample');
[idx2,R2] = kmeans(M2',representatives,'Distance','cosine','Start','sample');
[idx3,R3] = kmeans(M3',representatives,'Distance','cosine','Start','sample');
[idx4,R4] = kmeans(M4',representatives,'Distance','cosine','Start','sample');
[idx5,R5] = kmeans(M5',representatives,'Distance','cosine','Start','sample');
[idx6,R6] = kmeans(M6',representatives,'Distance','cosine','Start','sample');
[idx7,R7] = kmeans(M7',representatives,'Distance','cosine','Start','sample');
[idx8,R8] = kmeans(M8',representatives,'Distance','cosine','Start','sample');
[idx9,R9] = kmeans(M9',representatives,'Distance','cosine','Start','sample');
[idx10,R10] = kmeans(M10',representatives,'Distance','cosine','Start','sample');
[idx11,R11] = kmeans(M11',representatives,'Distance','cosine','Start','sample');
[idx12,R12] = kmeans(M12',representatives,'Distance','cosine','Start','sample');
[idx13,R13] = kmeans(M13',representatives,'Distance','cosine','Start','sample');
[idx14,R14] = kmeans(M14',representatives,'Distance','cosine','Start','sample');
[idx15,R15] = kmeans(M15',representatives,'Distance','cosine','Start','sample');
[idx16,R16] = kmeans(M16',representatives,'Distance','cosine','Start','sample');
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
% apply k-nearest neighbors
mdl = fitcknn([R1;R2;R3;R4;R5;R6;R7;R8;R9;R10;R11;R12;R13;R14;R15;R16],label2,'Distance','cosine') %kNN
% set number auf neighbors
mdl.NumNeighbors = 1;
mdl.Distance = 'cosine'
mdl.BreakTies = 'nearest'

% END PREDICT TEST DATA
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PLOT RESULTS
figure('Name',mdl.Distance);
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
        % set s to the label of the next representatives to actual dataset
        
        s = predict(mdl,Data(:,y)');
        % plot color and number at position of dataset depending on s
        if (s==1)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'r');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'13')
        elseif (s==2)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'b');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'9')
        elseif (s==3)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'g');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'5')
        elseif (s==4)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'y');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'1')
        elseif (s==5)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'm');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'14')
        elseif (s==6)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'c');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'10')
        elseif (s==7)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'k');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'6')
        elseif (s==8)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', [1 0.4 0.6]);
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'2')
        elseif (s==9)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'r');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'15')
        elseif (s==10)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'b');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'11')
        elseif (s==11)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'g');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'7')
        elseif (s==12)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'y');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'3')
        elseif (s==13)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'm');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'16')
        elseif (s==14)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'c');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'12')
        elseif (s==15)
            plot(Coord(y,1),Coord(y,2),'o', 'LineWidth', 2, 'MarkerSize', 18, 'Color', 'k');
            text((Coord(y,1)-xoffset),(Coord(y,2)+yoffset),'8')
        elseif (s==16)
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