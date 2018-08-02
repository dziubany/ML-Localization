function [R] = kMeansCOSY(M, k, mode)
    % k-Means classifier
    % Input:
    % M: Matrix of coloumnwise training data
    % k: Number of representatives to calculate
    % Mode: Distance options
    %   1  cosine
    %   2  euclidean 
    %   3  squared euclidean
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Output:
    % R: Matrix of k representatives
    % begin:
    % error handling, check if sizes matching
    attributes = length(M(1,:));
    nrSets = length(M(:,1));
    if nrSets < k
        msg = 'Error occurred. Not enough trainingsets to calculate k representatives';
        error(msg)
    end
    % create working Matrix for random trainingset selection
    Mwork = M;
    % set initial representatives
    Rnew = zeros(k, attributes);
    Rold = inf(k, attributes);
    representative = zeros(1, nrSets);
    % chosse k random representatives
    for i = 1 : k
        nrSets2 = length(Mwork(:,1));
        ran = randi(nrSets2);
        Rnew(i,:) = Mwork(ran,:);
        Mwork(ran,:) = [];
    end
    % set Rold ~= Rnew
    escape = 0;
    % as long as Rold ~= Rnew
    while escape == 0
        % find best match for each trainingset depending on cosine distance
        for i = 1 : nrSets
            best = inf;
            for j = 1 : k
                if mode == 1
                    % calculate cosine distance to all representatives 
                    tempProd1 = M(i,:)*Rnew(j,:)';
                    tempProd2 = M(i,:)*M(i,:)';
                    tempProd3 = Rnew(j,:)*Rnew(j,:)';
                    tempRoot1 = sqrt(tempProd2);
                    tempRoot2 = sqrt(tempProd3);
                    tempDenom = tempRoot1 * tempRoot2;
                    d = 1 - (tempProd1/tempDenom);
                elseif mode == 2
                    vd = M(i,:)' - Rnew(j,:)';
                    d = sqrt(vd' * vd);
                elseif mode == 3
                    vd = M(i,:)' - Rnew(j,:)';
                    d = vd' * vd;
                else
                    msg = 'Error occurred. Undefined mode number';
                    error(msg)
                end
                % set representative for actual trainingset
                if d < best
                    best = d;
                    representative(i) = j;
                end
            end
        end
       % set Rold to Rnew
       Rold = Rnew;
       % for each of the k representatives
       for i = 1 : k
           members = sum(representative == i);
           % set new Rnew
           if members ~= 0
               tempM = zeros(1, attributes);
               % calculate point with min distance to each trainingset to
               % represent
               for j = 1 : nrSets
                   if representative(j) == i
                       tempM(1,:) = tempM(1,:) + M(j,:);
                   end
               end
               Rnew(i,:) = tempM(1,:)/members;
           else
               Rnew(i,:) = Rnew(i,:);
           end
           tempM(1,:) = [];
           % calculate norm of Rnew
           Rnew(i,:) = Rnew(i,:)/norm(Rnew(i,:));
       end
       % iterate till no more changes for Rnew
       if Rold == Rnew
           escape = 1;
       end
    end
    % set output
    R = Rnew;
end