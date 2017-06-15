classdef SubVolumeSystem < ConeBeamSystem
    %Subclass of ConeBeamSystem that has to have associated translation
    %stage and objective stage objects. The calibration functions to find the
    %magnfication changes for the piezo and etl are found in this class.
    
    properties
        translationStage % TranslationStage class object
        objectiveStage % ObjectiveStage class object
    end
    
    properties (Access=private)
        objectiveR % effecitve objective source-detector distance
        piezoMagChange % MagChange object representing change in mag due to piezo alone
        etlMagChange % MagChange object representing change in mag due to piezo AND ETL
    end
    
    methods
        % CONSTRUCTOR
        % @param varargin - can either be 0 or 2 arguments:
        %        @param TranslationStage tStage, translation stage class
        %        @param ObjectiveStage oStage, piezo stage class
        function obj = SubVolumeSystem(varargin)
            obj = obj@ConeBeamSystem();
            nargin
            if nargin==0
                obj.translationStage = TranslationStage();
                obj.objectiveStage = ObjectiveStage();
            elseif nargin==2
                for i=1:nargin
                    switch class(varargin{i})
                        case 'TranslationStage'
                            obj.translationStage = varargin{i};
                        case 'ObjectiveStage'
                            obj.objectiveStage = varargin{i};
                    end
                end
            end
            
            if isempty(obj.translationStage) || isempty(obj.objectiveStage)
                error('Objective or Translation Stage unassigned');
            end
        end
        
        %GET/SET METHODS
        function out = getTransStage(obj)
            out = obj.translationStage;
        end
        function out = getObjectiveStage(obj)
            out = obj.objectiveStage;
        end
        % @param TranslationStage tStage, translation stage class object
        function setTransStage(obj,tStage)
            obj.translationStage = tStage;
        end
        % @param ObjectiveStage obStage, objective stage class object
        function setObjectiveStage(obj,obStage)
            obj.objectiveStage = obStage;
        end
        function out = getObjectiveR(obj)
            out = obj.objectiveR;
        end
        function out = getPiezoMagChange(obj)
            out = obj.piezoMagChange;
        end
        function out = getETLMagChange(obj)
            out = obj.etlMagChange;
        end
    end
    
    %% Load translation stage and piezo movements
    methods
        function loadTStage(obj)
            if ~isempty(obj.translationStage.getMagAoR)   
                if exist(fullfile(obj.getPath,'Stage Positions'),'file')==2
                    discretePositions = dlmread(fullfile(obj.getPath,'Stage Positions'));
                    dx = discretePositions(1,:);
                    dy = discretePositions(2,:);
                    
                    %% temporay code as images did not save with enough double string precision
                    dy = (dx-min(dx))/(max(dx)-min(dx))*(-1.806217+1.810204)+1.806217;
                    
                    %%
                    stepX = dx(2:end)-dx(1:end-1); stepX = nanmean(abs(stepX(stepX~=0)));
                    stepY = dy(2:end)-dy(1:end-1); stepY = nanmean(abs(stepY(stepY~=0)));
                    [xContinuousInfo,xFullMotion] = TranslationStage.fitSinusoidToDiscreteData(dx);
                    [yContinuousInfo,yFullMotion] = TranslationStage.fitSinusoidToDiscreteData(dy);
                    obj.translationStage.setMotionAngle(atan2(yContinuousInfo(1),xContinuousInfo(1)));
                    obj.translationStage.setMotion([dx-xContinuousInfo(3);dy-yContinuousInfo(3)]);
                    obj.translationStage.setFullMotion([xFullMotion-xContinuousInfo(3);yFullMotion-yContinuousInfo(3)]);
                    obj.translationStage.setStepSize([stepX;stepY]);
                else
                    error('No Stage Positions file exists');
                end
            else
                error('Please set magnification of translation stage movements and AoR first');
            end
        end
        
        function loadOStage(obj)
            if exist(fullfile(obj.getPath,'Piezo Positions'),'file')==2
                piezoPositions = dlmread(fullfile(obj.getPath,'Piezo Positions'));
                obj.objectiveStage.setMotion(piezoPositions.*obj.getRefractiveIndex);
            else
                error('No Stage Positions file exists');
            end
        end
        
    end
    
    %% Rotation of projections and initialisation of stepper motor from calibiration information
    methods
        function rotateProjections(obj)
            if isempty(obj.translationStage.getMotionAngle) || isempty(obj.translationStage.getAngle)
                error('Please assign translation stage motion angle, and translation stage angle first');
            end
            rotationAngle = obj.translationStage.getMotionAngle-obj.translationStage.getAngle;
            p = obj.getProjection(1);
            [xx,yy]= meshgrid(1:size(p,2),1:size(p,1));
            xxR = xx.*cos(rotationAngle)-yy.*sin(rotationAngle);
            yyR = yy.*cos(rotationAngle)+xx.*sin(rotationAngle);
            projArray = single(zeros(size(obj.getAllProjections)));
            switch obj.getUseGPU
                case 0
                    for idx=1:obj.getNProj
                        tic
                        p = interp2(obj.getProjection(idx),xxR,yyR);
                        p(isnan(p))=0;
                        projArray(:,:,idx)=p;
                       % if rem(idx,obj.getNProj/25)==0
                            disp(sprintf('rotation completion: %.0f%%',idx/obj.getNProj*100));
                       % end
                       toc
                    end
                case 1
                    xxR=gpuArray(xxR); yyR = gpuArray(yyR);
                    for idx=1:obj.getNProj
                        tic
                        p = gpuArray(obj.getProjection(idx));
                        out = interp2(p,xxR,yyR);
                        out(isnan(out))=0;
                        out = gather(out);
                        toc
                        tic
                        projArray(:,:,idx)=p;
                        disp(sprintf('rotation completion: %.0f%%',idx/obj.getNProj*100));
                        toc
                    end
            end
        end
        
        % @param int index, index of projection required
        function out = rotateProjection(obj,index)
            rotationAngle = obj.translationStage.getMotionAngle-obj.translationStage.getAngle;
            switch obj.getUseGPU
                case 0
                    tic
                    p = obj.getProjection(index);
                    [xx,yy]= meshgrid(1:size(p,2),1:size(p,1));
                    xxR = xx.*cos(rotationAngle)-yy.*sin(rotationAngle);
                    yyR = yy.*cos(rotationAngle)+xx.*sin(rotationAngle);
                    out = interp2(p,xxR,yyR);
                    out(isnan(out))=0;
                    toc
                case 1
                    tic
                    p = gpuArray(obj.getProjection(index));
                    [xx,yy]= meshgrid(gpuArray(1:size(p,2)),gpuArray(1:size(p,1)));
                    xxR = xx.*cos(rotationAngle)-yy.*sin(rotationAngle);
                    yyR = yy.*cos(rotationAngle)+xx.*sin(rotationAngle);
                    out = interp2(p,xxR,yyR);
                    out(isnan(out))=0;
                    out = gather(out);
                    toc
            end
        end
    end
    %% Calibration methods
    methods
        % calculates the magnifcation change between 0 and 180 paired
        % images of a fluorescent bead shifted over the field of view. uses
        % the magnification change to calculate the effective
        % source-detector distance that is introduced by using the
        % objective stage. This value is used to scale the raw projections
        % before using them in cone beam reconstruction.
        % @param string type, either 'piezo' or 'both' for piezo move only,
        % or piezo and etl move together
        % @param Objective objective, objective used
        function [xReal,yReal,xProgram,yProgram,zProgram] = magnificationCalibration(obj,objective,type,varargin)
            if strcmp(type,'both')
                if isempty(obj.objectiveR)
                    error('Please run magnificationCalibration with PIEZO type first');
                end
            end
            if nargin>3
                im_path = obj.objectiveStage.getCalibPath;
            else
                im_path = uigetdir();
                obj.objectiveStage.setCalibPath(im_path);
            end

            im_dir = dir(fullfile(im_path,'*.tif'));
            if isempty(im_dir)
                im_dir = dir(fullfile(im_path,'*.tiff'));
            end

            nProj = length(im_dir); imPortionSize = 60;

            for i=1:2:nProj
                disp(i/nProj*100);
                [images,x,y,z] = SubVolumeSystem.parsePairImagesInfo(im_dir,im_path,i); %parse image name to find programmed positions
                xProgram((i+1)/2,:) = x;
                yProgram((i+1)/2,:) = y;
                zProgram = z.*obj.getRefractiveIndex;
                if i==1; thres = nanmean(nanmean(images(:,:,1))); end
                
                for imageNo=1:2
                    [xOut,yOut] = OPTSystem.fastPeakFind(images(:,:,imageNo),1,thres);
                    xCurrentImage(1:length(xOut),imageNo) = xOut;
                    yCurrentImage(1:length(yOut),imageNo) = yOut;
                end
                if i==1
                    [xInitial,yInitial] = SubVolumeSystem.userSelectInitialPoints(images,xCurrentImage,yCurrentImage);
                    xReal((i+1)/2,1:2) = xInitial; yReal((i+1)/2,1:2) = yInitial;
                    sumImage = zeros(size(images));
                    obj.setBinFactor(1);
                else
                     [xReal,yReal] = SubVolumeSystem.findNextClosestPoint(xReal,yReal,xProgram,yProgram,xCurrentImage,yCurrentImage,objective.getMagnification,obj.getPixelSize,i);   
                end
               
                for imageNo=1:2
                    [imagePortion,idxX,idxY] = SubVolumeSystem.getImagePortion(images(:,:,imageNo),xReal((i+1)/2,imageNo),yReal((i+1)/2,imageNo),imPortionSize);
                    sumImage(idxY,idxX,imageNo) = imagePortion;
                end
                
                
            end

            [deltaMag,magChangeError,tform] = SubVolumeSystem.findMagnificationChange(xReal,yReal);
            movReg = imwarp(sumImage(:,:,1),tform,'outputview',imref2d([size(images,1),size(images,2)]));
            figure; subplot(1,2,1); imshowpair(sumImage(:,:,1),sumImage(:,:,2),'scaling','joint');
            title(sprintf('deltaM=%.4f',deltaMag));
            subplot(1,2,2); imshowpair(movReg,sumImage(:,:,2),'scaling','joint'); drawnow;
            title(sprintf('errdeltaM=%.6f',magChangeError));
            savefig(fullfile(im_path,sprintf('deltaM=%.4f,err=%.6f.fig',deltaMag,magChangeError)));
            
            switch type
                case 'piezo'
                    obj.piezoMagChange = MagChange(deltaMag,zProgram(1,1),zProgram(1,2),im_path);
                    obj.objectiveR = obj.piezoR(objective);
                    obj.findStageAngleandConstantMagnification(xReal,yReal,xProgram,yProgram,zProgram,objective);
                case 'both'
                    obj.etlMagChange = MagChange(deltaMag,zProgram(1,1),zProgram(1,2));
                    obj.setR(obj.etlWithPiezoR(objective));
                otherwise
                    error('Invalid type');
            end
        end   
        
        %calcuates translation stage angles of axes relative to pixel
        %sensors and magnifcation of stage movements at rotation axis
        function findStageAngleandConstantMagnification(obj,xReal,yReal,xProgram,yProgram,zProgram,objective)
            pxSz = obj.getPixelSize;
            magGuess = objective.getMagnification;
            for imageNo=1:2
                xP = (xProgram(:,imageNo)-mean(xProgram(:,imageNo)))./pxSz; yP = (yProgram(:,imageNo)-mean(yProgram(:,imageNo)))./pxSz;
                xR = xReal(:,imageNo)-mean(xReal(:,imageNo));
                yR = yReal(:,imageNo)-mean(yReal(:,imageNo));
                angleGuess = 0.01; %in radians
                
                [~,lc] = min(xR.*yR);
                xRmin = xR(lc);
                yRmin = yR(lc);

                startGuess = [magGuess,angleGuess];
                model = @minimiseError; modelRefine = @minimiseErrorWithOutliersRemoved;
                options = optimset('MaxFunEvals',500*length(startGuess),'MaxIter',500*length(startGuess));

                est(imageNo,:) = fminsearch(model,startGuess,options);
                est(imageNo,:) = fminsearch(modelRefine,est(imageNo,:),options);
            end
            
            magChangeAlternativeMethod = est(2,1)/est(1,1);
            ratio = (1-magChangeAlternativeMethod)/(1-obj.piezoMagChange.deltaMag);
            
            if ratio>1.1 || ratio<0.9
                error('Error fitting results in incorrect estimates of magnification change between images');
            end 
            
            objectiveRinMM = obj.objectiveR*pxSz/magGuess;
            zAoR = obj.objectiveStage.getAoRLocation;
            magAoR = est(1,1)/(1+zAoR/objectiveRinMM)*(1+zProgram(1)/objectiveRinMM);
            
            obj.translationStage.setAngle((est(2,2)+est(1,2))/2);
            obj.translationStage.setMagAoR(magAoR);
            
            function [sse] = minimiseError(params)
                mag = params(1);
                angle = params(2);

                x2 = (xP.*cos(angle)-yP.*sin(angle)).*mag;
                y2 = (xP.*sin(angle)+yP.*cos(angle)).*mag;
                [~,lc] = min(x2.*y2);
                x2 = x2-x2(lc)+xRmin;
                y2 = y2-y2(lc)+yRmin;
                
                for i=1:length(x2)
                    r = sqrt((x2(i)-xR).^2+(y2(i)-yR).^2);
                    [mn,lc] = min(r);
                    mins(i) = mn;
                end              
                sse = nansum(nansum(mins));
            end

            function [sse,r] = minimiseErrorWithOutliersRemoved(params)
                mag = params(1);
                angle = params(2);

                x2 = (xP.*cos(angle)-yP.*sin(angle)).*mag;
                y2 = (xP.*sin(angle)+yP.*cos(angle)).*mag;
                [~,lc] = min(x2.*y2);
                x2 = x2-x2(lc)+xRmin;
                y2 = y2-y2(lc)+yRmin;
                
                for i=1:length(x2)
                    r = sqrt((x2(i)-xR).^2+(y2(i)-yR).^2);
                    [mn,lc] = min(r);
                    mins(i) = mn;
                end
                mins(mins>100)=NaN;
                sse = nansum(nansum(mins));

                %scatter(x2,y2); hold on; scatter(xR,yR); hold off;
                %drawnow;
            end

        end

    end

    %% private methods
    methods 
         %calculate effective source-detector distance R in unbinned pixels
        %for piezo only moving
        % @param MagChange piezoMagChange
        % @param OPTSystem optSys, OPTSystem object
        % @param Objective objective, Objective object
        function out = piezoR(obj,objective)
            if isempty(obj.piezoMagChange)
                error('Please run piezo magcalibration first');
            else
                zAoR = (obj.piezoMagChange.z2+obj.piezoMagChange.z1)/2;
                dz = (obj.piezoMagChange.z2-obj.piezoMagChange.z1)/2;
                obj.objectiveStage.setAoRLocation(zAoR);

                R_p = -zAoR+dz*(1+obj.piezoMagChange.deltaMag)/(1-obj.piezoMagChange.deltaMag);

                out = R_p/(obj.getPixelSize/obj.getBinFactor)*objective.getMagnification;

                obj.objectiveR = out;
            end
        end
        
        %calculate effective source-detector distance R in unbinned pixels
        %for piezo AND etl only moving
        % @param MagChange piezoMagChange
        % @param OPTSystem optSys, OPTSystem object
        % @param Objective objective, Objective object
        function out = etlWithPiezoR(obj,objective)
            if isempty(obj.piezoMagChange) ||isempty(obj.etlMagChange) 
                error('Please run piezo mag and normal mag calibration first');
            else
                 Rp = obj.piezoR(objective)*(obj.getPixelSize/obj.getBinFactor)/objective.getMagnification;
                 dMe = obj.etlMagChange.deltaMag;
                 zp1 = obj.piezoMagChange.z1;
                 zp2 = obj.piezoMagChange.z2;
                 z1 = obj.etlMagChange.z1;
                 z2 = obj.etlMagChange.z2;
                 R_e = ((1+zp1/Rp)*(z1-zp1)-dMe*(1+zp2/Rp)*(z2-zp2))/(dMe*(1+zp2/Rp)-(1+zp1/Rp));
                 out = R_e/(obj.getPixelSize/obj.getBinFactor)*objective.getMagnification;
            end
        end
    end
    
    %% static methods
    methods (Static)
        % parses information from 180 degree paired images. These are
        % images of a fluorescent beads at angle 0 and 180 degree such that
        % they should overlap in a perfect standard 4f system (i.e closest
        % and furthest position from the objective). 
        % @param struct im_dir, directory of images
        % @param string im_path, path to folder containing images
        % @param int idx, index of image to load
        function [image,x,y,z] = parsePairImagesInfo(im_dir,im_path,idx)
                info1 = strsplit(im_dir(idx).name,{'x=','z=','y=','ETL=','mA','.t'}); %string information from image 1
                info2 = strsplit(im_dir(idx+1).name,{'x=','z=','y=','ETL=','mA','.t'}); %string information from image 2

                % the first image is taken at a lower z depth
                if str2num(cell2mat(info2(4)))>str2num(cell2mat(info1(4)))
                    loadIdx = [idx,idx+1];
                else
                    loadIdx = [idx+1,idx];
                end
                
                for k=1:2
                    info = strsplit(im_dir(loadIdx(k)).name,{'x=','z=','y=','ETL=','mA','.t'}); %string information from image k (unfortuanely z/y are switched here!!!)
                    x(1,k) = str2num(cell2mat(info(2))); %programmed translation stage x location
                    y(1,k) = str2num(cell2mat(info(3))); %programmed translation stage y location
                    z(1,k) = str2num(cell2mat(info(4))); %programmed objective stage z location
                    image(:,:,k) = single(imread(strcat(im_path,'\',char(cellstr(im_dir(loadIdx(k)).name)))));
                end
        end
        
        % displays first pair of images and asks user to enter point
        % location that they want to track for the rest of the images. (i.e
        % the location of the bead they have found the 0 and 180 degree
        % pair of)
        % @param double[][][] images, 2 paired images
        % @param double[][] xReal, x-location of peaks in both images
        % @param double[][] yReal, y-location of peaks in both images
        % @return double[1x2] xInitial, x-location of desired peak in both
        % images
        % @return double[1x2] yInitial, y-location of desired peak in both
        % images
        function [xInitial,yInitial] = userSelectInitialPoints(images,xReal,yReal)
            figure;
            for imageNo=1:2
                imagesc(images(:,:,imageNo)); hold on; scatter(xReal(:,imageNo),yReal(:,imageNo),'rx'); hold off;
                x = input('enter Red x-loc: ');
                y = input('enter Red y-loc: ');
                r = sqrt((xReal(:,imageNo)-x).^2+(yReal(:,imageNo)-y).^2);
                [~,lc] = min(r);
                xInitial(1,imageNo) = xReal(lc,imageNo);
                yInitial(1,imageNo) = yReal(lc,imageNo);
            end
        end
        
        % finds portion of image and returns index locations so can be
        % easily inserted into a larger image. Does not return anything
        % beyond image limits
        % @param double[][] images, single image
        % @param double xLocation, x-location of desired peak
        % @param double yLocation, y-location of desired peak 
        % @param double[1xlength] size, desired length of portion
        function [imagePortion,idxX,idxY] = getImagePortion(image,xLocation,yLocation,portionSize)
            if ~isempty(xLocation)
                idxY = yLocation-portionSize/2:yLocation+portionSize/2; 
                idxY(or(idxY<1,idxY>size(image,1))) = NaN;
                idxX = xLocation-portionSize/2:xLocation+portionSize/2; 
                idxX(or(idxX<1,idxX>size(image,2))) = NaN;
                idxY = idxY(~isnan(idxY)); idxX = idxX(~isnan(idxX));
                imagePortion = image(idxY,idxX)./max(max(image(idxY,idxX)));
            else
                imagePortion = []; idxX = []; idxY = [];
            end
        end
        
        % finds next real closest peak to programmed value
        % @param double[][] xReal/yReal, previous real locations of tracked
        % peak in pixels
        % @param double[][] xProgram/yProgram, programmed stage locations
        % in mm
        % @param double[] xCurrentImage/yCurrentImage, location of all
        % peaks in current image in pixels
        % @param double magGuess, guess of magnifcation from objective
        % @param double pxSz, binned pixel size of images in mm
        % @param int idx, index of current pair images
        % @return double[][] xReal,yReal, updated real location of tracked beads
        function [xReal,yReal] = findNextClosestPoint(xReal,yReal,xProgram,yProgram,xCurrentImage,yCurrentImage,magGuess,pxSz,idx)
            for k=1:2
                xPred = -1*(xProgram((idx+1)/2,k)-xProgram((idx+1)/2-1,k))./pxSz*magGuess+xReal((idx+1)/2-1,k);
                yPred = -1*(yProgram((idx+1)/2,k)-yProgram((idx+1)/2-1,k))./pxSz*magGuess+yReal((idx+1)/2-1,k);
                r = sqrt((xCurrentImage(:,k)-xPred).^2+(yCurrentImage(:,k)-yPred).^2);
                [~,minLocation] = min(r);
                xReal((idx+1)/2,k) = xCurrentImage(minLocation,k);
                yReal((idx+1)/2,k) = yCurrentImage(minLocation,k);
            end
        end
        
        
        
        %fitting function between sets of equivalent points, that are
        %translated and scaled with respect to each other
        % @param double[][] xReal, [nPoints x 2] vector of x-locations
        % @param double[][] yReal, [nPoints x 2] vector of y-locations
        function [magChange,magChangeError,tform] = findMagnificationChange(xReal,yReal)
            r = sqrt((xReal(1,2)-xReal(:,1)).^2+(yReal(1,2)-yReal(:,1)).^2);
            [mn,lc] = min(r);
            xGuessFirst = xReal(lc,1); yGuessFirst = yReal(lc,1);
            
            r = sqrt((xReal(end,2)-xReal(:,1)).^2+(yReal(end,2)-yReal(:,1)).^2);
            [mn,lc] = min(r);
            xGuessEnd = xReal(lc,1); yGuessEnd = yReal(lc,1);

            aGuess = (xReal(end,2)-xReal(1,2))./(xGuessEnd-xGuessFirst);
            eGuess = xReal(1,2)-xGuessFirst.*aGuess;
            fGuess = yReal(1,2)-yGuessFirst.*aGuess;

            figure;
            startGuess = [aGuess,eGuess,fGuess];
            model = @fun;
            options = optimset('MaxFunEvals',500*length(startGuess),'MaxIter',500*length(startGuess));
            est = fminsearch(model,startGuess,options);
            [~,xdatOut,ydatOut,xfitOut,yfitOut] = model(est);

            x = xReal(:,2);
            y = yReal(:,2);
            
            r2 = sqrt((x-xfitOut).^2+(y-yfitOut).^2);
            
            x = xReal(:,1);
            y = yReal(:,1);

            ciX = confint(fit(x(r2<50),xdatOut(r2<50),'poly1'));
            X = coeffvalues(fit(x(r2<50),xdatOut(r2<50),'poly1'));
            ciY = confint(fit(y(r2<50),ydatOut(r2<50),'poly1'));
            Y = coeffvalues(fit(y(r2<50),ydatOut(r2<50),'poly1'));
            erX = (abs(ciX(1,1)-X(1))+abs(ciX(2,1)-X(1)))/2;
            erY = (abs(ciY(1,1)-Y(1))+abs(ciY(2,1)-Y(1)))/2;

            magChangeError = sqrt(erX.^2+erY.^2);
                
                
            tform = affine2d([est(1),0,0;0,est(1),0;est(2),est(3),1]);
            magChange = tform.T(1,1);

            function [sse,xdat,ydat,xfit,yfit] = fun(params)
                a = params(1);
                e = params(2);
                f = params(3);

                xfit = xReal(:,1).*a+e;
                yfit = yReal(:,1).*a+f;

                for i=1:length(xfit)
                    r = sqrt((xfit(i)-xReal(:,2)).^2+(yfit(i)-yReal(:,2)).^2);
                    [mn,lc] = min(r);
                    mins(i) = mn;
                    xdat(i,1) = xReal(lc,2);
                    ydat(i,1) = yReal(lc,2);
                end
                sse = nansum(mins);
            end
        end
        


        
       
    end
    
    
        
    
end

