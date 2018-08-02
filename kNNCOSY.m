function [labelX0] = kNNCOSY(M, labelM, k, X0, mode)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % k-Nearest Neighbor classifier
    % Input:
    % M: Matrix of coloumnwise training data
    % labelM: Labels for training data
    % k: Number of nearest neighbors to classify
    % X0: Matrix of Testsets
    % Mode: Distance options
    %   1  cosine
    %   2  euclidean 
    %   3  squared euclidean
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Output:
    % labelX0: Predicted labels of X0
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % begin:
    % error handling, check if sizes matching
    lenlabelM = length(labelM);
    lenM = length(M(:,1));
    
    datapointsX0 = length(X0(:,1));
    datapointsM = length(M(1,:));

    if lenM ~= lenlabelM
        msg = 'Error occurred. Length of trainingsets mismatch length label';
        error(msg)
    end
    
    if datapointsX0 ~= datapointsM
        msg = 'Error occurred. Length of Test dataset mismatch length trainingsets';
        error(msg)
    end
    
    if k > lenM
        msg = 'Error occurred. Not enough trainingsets to find k neighbors';
        error(msg)
    end
    
    % prepare labelX0
    lenX0 = length(X0(1,:));
    labelX0 = zeros(1, lenX0);
    
    % prepare array for algorithm
    distk = inf(1, k);
    labelk = zeros(1, k);
    
    % on each testset,
    for test = 1 : lenX0
        % check for each trainingset,
        for i = 1 : lenM;
             % distance between testset and traingset
             if mode == 1
                 tempProd1 = M(i,:)*X0(:,test);
                 tempProd2 = M(i,:)*M(i,:)';
                 tempProd3 = X0(:,test)'*X0(:,test);
                 tempRoot1 = sqrt(tempProd2);
                 tempRoot2 = sqrt(tempProd3);
                 tempDenom = tempRoot1 * tempRoot2;
                 d = 1 - (tempProd1/tempDenom);
             elseif mode == 2
                 vd = M(i,:)' - X0(:,test);
                 d = sqrt(vd' * vd);
             elseif mode == 3
                 vd = M(i,:)' - X0(:,test);
                 d = vd' * vd;
             else
                 msg = 'Error occurred. Undefined mode number';
                 error(msg)
             end
             % sort distances and labels, smallest distance first
             j = 1;
             while j < k + 1
                if d < distk(j)
                    for p = k : -1 : j + 1
                        distk(p) = distk(p-1);
                        labelk(p) = labelk(p-1);
                    end
                    distk(j) = d;
                    labelk(j) = labelM(i);
                    j = k + 1;
                else
                    j = j + 1;
                end
             end
        end
        % check for shortest distance between all most often in labelk
        % get most occurences
        list = unique(labelk);
        [n, index] = histc(labelk, list); 
        z = max(n);
        y = list(n == z);
        % get shortest distance of most occurences
        [sharedVals,idxsIntoA] = intersect(labelk,y);
        labelX0(test) = labelk(min(idxsIntoA));    
    end
end