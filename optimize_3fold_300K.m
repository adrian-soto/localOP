% Classification with 3-fold partition of data:
% validation, training and test  

%% -1 to HD
%% +1 to LD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Flags to enable different actions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Actions

Temp          = '300K'
T             = 300

ReadDData     = true;

setTrainData  = true;

trainGaussSVM = true;

trainData    = 'LSI_IPES_hardcut'; 

doCorr       = true;
doCanCorr    = false;
corrcoefsfile='corr.dat'

doPCA        = false;
doPCApp      = false;
doPCAvsT     = false;

doTTM3F_IPES = true;
SVMonTTM3F   = true;

doTempPlots  = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data subsampling
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ssdata       = true;
ssFreeup     = true;       % free up memory by overwriting the arrays with the subsampled onesss
Ns_tra       = 20000                      % Number of samples for training
Ns_val       = 20000                      % Number of samples for validation
Ns_tes       = 20000                      % Number of samples for test
Ns_tot       = Ns_val+Ns_tra+Ns_tes;     % Total number of samples
is_val       = 1:Ns_val;                 % Indices for validation
is_tra       = Ns_val+(1:Ns_tra);        % Indices for training
is_tes       = Ns_val+Ns_tra+(1:Ns_tes); % Indices for classification
is_tot       = 1:Ns_tot;

fiftyHDfiftyLD = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     FEATURE INDICES
% 1:5   : Voronoi volume, surface, numbers of vertices, edges and sides
% 6:7   : q, Sk, 
% 8:20  : LSI0, LsI1, ... LSI12
% 21:37 : 17 O-O distances
% 38:42 : 5 O-H-H angles
% 43:48 : 6 O-O-O angles
% 49:50 : Nacc, Ndon
% 51:63 : Number of H-bonds loops of length 3, 4, ..., 15
iftr          = [1:8, 21:58] %1:63;   % indices of features to be used
iftrlabel     ='[1:8, 21:58]'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% SVM options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
EvaluateError = true;
trainMLPSVM   = false;
SVMoptions = statset('MaxIter',100000);
CrossValSVM   = false;

boxC_all      = [0.5, 2.0, 4.00];
rbfsigma_all  = [0.5, 2.0, 4.0, 8.0];
OptParamsFile = 'SVMOptParams.dat'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    P R O G R A M   B E G I N S   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Import data
if ReadDData == true
   
    disp('Reading preprocessed data ...')
    
    datadir='./data'

    D_dat=strcat(datadir, '/D300K.dat');

    D=importdata(D_dat, ' ', 0);
    
    
    Ntot = size(D,1);
     
    if(Ntot < Ns_tot)
        disp('ERROR: number of data points is smaller than number of requested samples. EXITING...')
        exit
    end

    
    

    if setTrainData == true

        if strcmp(trainData, 'LSI_IPES_hardcut')
            % Select data with  LSI < LSIleft and LSI >= LSIright
            disp('Setting training data == LSI_IPES_hardcut')
        
            iLSI    = 8 % Index to column containing LSI
            itarget = 1
            iftr = iftr + 1 % Since first index in D contains target value, shift by one  

        
        
            %%%%%%%%%%%%%%%%%%%%%%%%%%
            % aggregate coordinates from all temperatures
        
             % Validation data
            x=D(:,iftr); %
        
            t=D(:,itarget);%
            
            if fiftyHDfiftyLD == true
	      %
	      % Balance training and validation 
	      % sets but NOT test set.
	      %
	      % Algorithm steps
	      % 1) Create array iall=[1, 2, ... , NumberOfData]
	      %
	      % 2) Shuffle it in place
	      %
	      % 3) Sample first Ns_tes points for test set
	      %
	      % 4) Balance remaining points. Sample Ns_tra and Ns_val
	      %  4.1) Find indices to HD and LD points
	      %  4.2) Take all points from smaller group and
	      %       the same number from bigger group. Put 
	      %       all of them into a single index array.
	      %  4.3) Shuffle it in place so that HD and LD
	      %       points are mixed up
	      %  4.4) Sample indices for training and validation
	      %
	      % 5) Sample data
	      %
	      
	      
	      % 1) and 2)
	      Nall=length(t);
	      iall=randperm(Nall);
	      
	      % 3) 
	      ishuff_tes=iall(1:Ns_tra);
	      
	      % 4)
	      
	      % 4.1)
	      iHDall = find(t == -1);
	      iLDall = find(t == +1);
	      
	      % find remaining indices (not already sampled)
	      irem = iall(Ns_tra+1:Nall);
	      % from these, find indices corresponding to HD and LD
	      iHDrem=[];
	      iLDrem=[];
	      for i = irem
		if (t(i) == -1)
		  iHDrem = [iHDrem, i];
		elseif (t(i) == +1)
		  iLDrem = [iLDrem, i];
		else
		  disp('ERROR! t(i) not equal to -1 or +1. Exiting ...')
		  exit()
		  end
		end
		  
		% 4.2)
		Ns_aux = min(length(iHDrem), length(iLDrem));
		iaux = [iHDrem(1:Ns_aux), iLDrem(1:Ns_aux)];
		% 4.3) 
		iaux = iaux(randperm(length(iaux)));
		
		% 4.4)
		ishuff_tra = iaux(1:Ns_tra);
		ishuff_val = iaux(Ns_tra+1:Ns_tra+Ns_val);
		
		clear iaux iall iHDall iLDall

            else
                % Sample uniformly from input data
                ishuff=randperm(Ntot);
		
		% Generate indices to shuffle data points
	        ishuff_val=ishuff(is_val);
            	ishuff_tra=ishuff(is_tra);
            	ishuff_tes=ishuff(is_tes); 

            end
            
            
            % Validation data
            x_val=x(ishuff_val,:);
            t_val=t(ishuff_val,:);
        
            % Training data
            x_tra=x(ishuff_tra,:);
            t_tra=t(ishuff_tra,:);
  
            % Test data
            x_tes=x(ishuff_tes,:);
            t_tes=t(ishuff_tes,:);

      
        
        
            % collect data into arrays for validation, training and test
            D_val=cat(2, t_val, x_val); 
            D_tra=cat(2, t_tra, x_tra); 
            D_tes=cat(2, t_tes, x_tes);
            % In D_ arrays, 1st column is t, and
            % adjacent columns are the features.
            itargetD= 1
            iftrD   = (1:length(iftr)) + 1       
                
        end
    
    end

end


% Calculate correlation coefficients of training data
if doCorr == true
    
    disp('Calculating correlation coefficients ...')

    [R_P, P_P] = corr(D_tra(:,itargetD), D_tra(:,iftrD), 'type', 'Pearson');
    [R_S, P_S] = corr(D_tra(:,itargetD), D_tra(:,iftrD), 'type', 'Spearman');
    [R_K, P_K] = corr(D_tra(:,itargetD), D_tra(:,iftrD), 'type', 'Kendall');

    % Collect all correlation coefficients and p-values and print
    ind=1:length(R_P); 
    disp('[feature index  R_P      P_P        R_S        P_S        R_K        P_K]')
    corrcoefs=[ind', R_P', P_P', R_S', P_S', R_K', P_K']
    
    save(corrcoefsfile,'corrcoefs', '-ascii');
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Validation over SVM with RBF kernel
% 1) Vary parameters over validation set.
% 2) Train on training set and calculate error
% 3) Run on test set
%

if trainGaussSVM == true

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Grid search training/validation sets
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    SVMparams=zeros(length(boxC_all)*length(rbfsigma_all), 2);
    ip=1;
    for iC = 1:length(boxC_all)
        Caux=boxC_all(iC);
        for isigma = 1:length(rbfsigma_all)
            SVMparams(ip,1)=Caux;
            SVMparams(ip,2)=rbfsigma_all(isigma);
            ip=ip+1;
        end
    end
    
    N_boxC=length(boxC_all);
    N_rbfsigma=length(rbfsigma_all);

    Cs=[]; % to store parameters [boxC, rbfsigma]
    
    
    tra_errors=[];%zeros(length(boxC_all)*length(rbfsigma_all), 1);
    tra_LDrelerrors=[];
    tra_HDrelerrors=[];
    SVM_tra_all=[];
    B_all_tra=[];
    O_all_tra=[];
    
    val_errors=[];%zeros(length(boxC_all)*length(rbfsigma_all), 1);
    val_LDrelerrors=[];
    val_HDrelerrors=[];
    SVM_tra_all=[];
    B_all_val=[];
    O_all_val=[];
    
    disp('Training SVM with Gaussian kernel ...')  
    for ip = 1:size(SVMparams,1)

        boxC=SVMparams(ip,1)
        rbfsigma=SVMparams(ip,2)
            
        SVMfile=strcat('GaussSVM_', num2str(boxC), '_', num2str(rbfsigma), '.mat')
        
        if (exist(SVMfile, 'file') == 2)
            disp('Reading pretrained SVM from file... At your own risk!!') 
            load(SVMfile, 'GaussSVM')
            
        else
            disp('Training SVM with Gaussian kernel ...') 
            GaussSVM=svmtrain(x_tra, t_tra, 'kernel_function', 'rbf', ...
                'rbf_sigma', rbfsigma, 'boxconstraint', boxC, 'Options', SVMoptions )
    
            save(SVMfile,'GaussSVM');
            
        end
            
        SVM_tra_all=[SVM_tra_all, GaussSVM];
            
        % Evaluate error on training set
        y_tra=[];
        for i=1:Ns_tra
            y_aux=svmclassify(GaussSVM, x_tra(i,:));
            y_tra=[y_tra; y_aux];
        end
    
        % Relative classification errors for total, LD and HD
        ypt_tra=y_tra+t_tra; % +2 -> correct LD, -2 -> correct HD, 0 - > total wrong
        ymt_tra=y_tra-t_tra; % +2 -> misscl. HD, -2 -> misscl. LD, 0 - > total right
        NtHD_tra=sum(t_tra==-1);  % target HD population
        NtLD_tra=Ns_tra-NtHD_tra; % target LD population
        %%%NyHD_tra=sum(y_tra==-1); NyLD_tra=Ns_tra-NyHD_tra; % classification populations
    
        err_tra=sum(ypt_tra==0)/Ns_tra
        relerrHD_tra=sum(ymt_tra==+2)/NtHD_tra
        relerrLD_tra=sum(ymt_tra==-2)/NtLD_tra
        B_tra=relerrLD_tra-relerrHD_tra
        O_tra=err_tra %sqrt(err_tra^2 + B_tra^2)
        %%%relerrHD_tra=sum(ymt_tra==+2)/Ns_tra
        %%%relerrLD_tra=sum(ymt_tra==-2)/Ns_tra
        
        tra_errors=[tra_errors; err_tra];
        tra_HDrelerrors=[tra_HDrelerrors; relerrHD_tra];
        tra_LDrelerrors=[tra_LDrelerrors; relerrLD_tra];
        B_all_tra=[B_all_tra; B_tra];
        O_all_tra=[O_all_tra; O_tra];
            
            

    %if EvaluateError == true
        disp('Evaluating error on validation set SVM ...') 
        
        %Ns=size(D_val,1);
        y_val=[];
        for i=1:Ns_val
            y_aux=svmclassify(GaussSVM, x_val(i,:));
            y_val=[y_val; y_aux];
        end
        
        % Relative classification errors for total, LD and HD
        ypt_val=y_val+t_val; % +2 -> correct LD, -2 -> correct HD, 0 - > total wrong
        ymt_val=y_val-t_val; % +2 -> misscl. HD, -2 -> misscl. LD, 0 - > total right
        NtHD_val=sum(t_val==-1);  % target HD population
        NtLD_val=Ns_val-NtHD_val; % target LD population
        %%%NyHD_val=sum(y_val==-1); NyLD_val=Ns_val-NyHD_val; % classification populations
    
        err_val=sum(ypt_val==0)/Ns_val
        relerrHD_val=sum(ymt_val==+2)/NtHD_val
        relerrLD_val=sum(ymt_val==-2)/NtLD_val
        B_val=relerrLD_val-relerrHD_val
        O_val=err_val %sqrt(err_val^2 + B_val^2)
        %%%relerrHD_val=sum(ymt_val==+2)/Ns_val
        %%%relerrLD_val=sum(ymt_val==-2)/Ns_val
            
        val_errors=[val_errors; err_val];
        val_HDrelerrors=[val_HDrelerrors; relerrHD_val];
        val_LDrelerrors=[val_LDrelerrors; relerrLD_val];
        B_all_val=[B_all_val; B_val];
        O_all_val=[O_all_val; O_val];
            
    end
    
    
    % Plot errors
    figure; hold on
    plot(tra_errors)
    plot(tra_HDrelerrors)
    plot(tra_LDrelerrors)
    legend({'r', 'HD', 'LD'})
    title(strcat('Error rates on training set. iftr=',iftrlabel))
    savePDF('TrainErrors_tra.pdf');
    
    
    figure; hold on
    plot(val_errors)
    plot(val_HDrelerrors)
    plot(val_LDrelerrors)
    legend({'r', 'HD', 'LD'})
    title(strcat('Error rates on validation set. iftr=',iftrlabel))
    savePDF('TrainErrors_val.pdf');
    
    % Plot error measures
    figure; hold on
    plot(tra_errors)
    plot(B_all_tra)
    plot(O_all_tra)
    legend({'r', 'B', 'O'})
    title(strcat('Error measures on training set. iftr=',iftrlabel))
    %plot(zeros(1,length(val_errors)), 'color', 'k')
    savePDF('TrainErrMes_tra.pdf');
    
    figure; hold on
    plot(val_errors)
    plot(B_all_val)
    plot(O_all_val)
    legend({'r', 'B', 'O'})
    title(strcat('Error measures on validation set. iftr=', iftrlabel))
    %plot(zeros(1,length(val_errors)), 'color', 'k')
    savePDF('TrainErrMes_val.pdf');
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Training with parameter values that minimized O on validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    imin    = find(O_all_val==min(O_all_val));    
    if length(imin > 1)
        disp('W A R N I N G !!! ')
        disp('2 or more parameter pairs give te same O.')
        disp('Setting imin to the first pair...')
        imin=imin(1);
    end
    
    % Retrieve values
    boxC     = SVMparams(imin,1);
    rbfsigma = SVMparams(imin,2);
    disp('The values that minimize O are ')
    SVMOptParams=[boxC, rbfsigma]

    % Save parameters to file
    save(OptParamsFile,'SVMOptParams', '-ascii');
   
    
    % Retrain SVM on training and validation sets together
    x_tv=cat(1, x_tra, x_val);
    t_tv=cat(1, t_tra, t_val);
    Ns_tv=length(t_tv)
    
    GaussSVM=svmtrain(x_tv, t_tv, 'kernel_function', 'rbf', ...
                'rbf_sigma', rbfsigma, 'boxconstraint', boxC, 'Options', SVMoptions )
            
    %save('GaussSVMtv.mat','GaussSVM');      
            
            
    disp('Evaluating error on tv set SVM ...') 
    y_tv=[];
    for i=1:Ns_tv
        y_aux=svmclassify(GaussSVM, x_tv(i,:));
        y_tv=[y_tv; y_aux];
    end
    
    
    % Relative classification errors for total, LD and HD
    ypt_tv=y_tv+t_tv; % +2 -> correct LD, -2 -> correct HD, 0 - > total wrong
    ymt_tv=y_tv-t_tv; % +2 -> misscl. HD, -2 -> misscl. LD, 0 - > total right
    NtHD_tv=sum(t_tv==-1); NtLD_tv=Ns_tv-NtHD_tv; % target LD and HD populations
    %NyHD_tv=sum(y_tv==-1); NyLD_tv=Ns_tv-NyHD_tv; % classification populations
    
    err_tv=sum(ypt_tv==0)/Ns_tv
    relerrHD_tv=sum(ymt_tv == +2)/NtHD_tv
    relerrLD_tv=sum(ymt_tv == -2)/NtLD_tv
    %relerrHD_tv=sum(ymt_tv == +2)/Ns_tv
    %relerrLD_tv=sum(ymt_tv == -2)/Ns_tv
    B_tv=relerrLD_tv-relerrHD_tv
    O_tv=err_tv %sqrt(err_tv^2 + B_tv^2)
    
    
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Evaluation on test set
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('Evaluating error on test set SVM ...')
    y_tes=[];
    for i=1:Ns_tes
        y_aux=svmclassify(GaussSVM, x_tes(i,:));
        y_tes=[y_tes; y_aux];
    end        
    
    % Relative classification errors for total, LD and HD
    ypt_tes=y_tes+t_tes; % +2 -> correct LD, -2 -> correct HD, 0 - > total wrong
    ymt_tes=y_tes-t_tes; % +2 -> misscl. HD, -2 -> misscl. LD, 0 - > total right
    NtHD_tes=sum(t_tes==-1); NtLD_tes=Ns_tes-NtHD_tes; % target LD and HD populations
    %NyHD_tes=sum(y_tes==-1); NyLD_tes=Ns_tes-NyHD_tes; % classification populations
    
    err_tes=sum(ypt_tes==0)/Ns_tes
    relerrHD_tes=sum(ymt_tes == +2)/NtHD_tes
    relerrLD_tes=sum(ymt_tes == -2)/NtLD_tes
    B_tes=relerrLD_tes-relerrHD_tes
    O_tes=err_tes %sqrt(err_tes^2 + B_tes^2)
    

end
