classdef (Abstract) OPTSystem < handle & matlab.mixin.Copyable
    %Class defining an optical projection system and its properties.
    %Coordinate system is by default centred on motor axis.
    %
    
    properties (Access = private)
        % Experimental Parameters
        nProj %number of projections
        nAngles %number of angles (either 180 or 360)
        opticCentre = [0,0] %(x,y) location of optic centre in object space mm
        rotationDirection  %defines rotation direction, either 'clock' or 'anti'
        axisDirection  %define rotation axis direction, either 'horiz' or 'vert'
        pxSz = 6.5e-3 %size of sensor pixels in mm (6.5e-3 by default)
        lambda = 500e-6; %wavelength in mm
        apertureRadius %radius of aperture stop in mm
        focalPlaneOffset % focal plane z-offset from motor axis in mm
        n = 1; %refractive index of immersion medium, default air=1
        
        % Large Data
        projections %raw projection information
        filteredProj = [] % filtered projections
        MIP = [] %maximum intensity projection through all projection angles
        
        % Computational Parameteres
        filter = 'ram-lak'; %filter used in reconstruction (Ram-Lak default)
        useGPU = 0; %boolean: true -> use CUDA enabled GPU
        interptype = 'linear'; %interpolateion type, set linear default
        path %path of raw projections
        outputPath %path of reconstructions
        binFactor = 1; %value of bin factor
        unbinnedWidth %number of pixels in direction perpendicular to the rotation axis (not binned)
        unbinnedHeight %number of pixels in direction parallel to the rotation axis (not binned)
    end
    
    properties (Access = public)
        % Simulation Parameters
        AF % AmbiguityFunction object
    end
    
    %% Constructor and get/set methods
    methods
        %constructor
        function obj = OPTSystem()
            obj.path = uigetdir();
        end           
        function out = getPath(obj)
            out = obj.path;
        end        
        function out = getAllProjections(obj)
            out = obj.projections;
        end
        function out = getUseGPU(obj)
            out = obj.useGPU;
        end
        %@param index, projection number
        function out = getProjection(obj,index)
            out = obj.projections(:,:,index);
        end  
        function out = getAxisDirection(obj)
            out = obj.axisDirection;
        end
        function out = getRotationDirection(obj)
            out = obj.rotationDirection;
        end
        function out = getWidth(obj)
            out = obj.unbinnedWidth/obj.binFactor;
        end        
        function out = getHeight(obj)
            out = obj.unbinnedHeight/obj.binFactor;
        end
        function out = getNAngles(obj)
            out = obj.nAngles;
        end
        function out = getNProj(obj)
            out = obj.nProj;
        end
        % gets the optic centre in image space binned pixels
        function out = getOpticCentrePixels(obj,objective)
            out = obj.opticCentre/obj.getPixelSize*objective.getMagnification;
        end
        % gets the optic centre in object space mm
        function out = getOpticCentre(obj)
            out = obj.opticCentre;
        end
        function out = getPixelSize(obj)
            out = obj.pxSz*obj.binFactor;
        end
        function out = getMotorAxisXDisplacement(obj)
            out = obj.motorAxisXDisplacement/obj.binFactor;
        end
        function out = getMotorAxisZDisplacement(obj)
            out = obj.motorAxisZDisplacement/obj.binFactor;
        end
        function out = getApertureRadius(obj)
            out = obj.apertureRadius;
        end
        function out = getLambda(obj)
            out = obj.lambda;
        end
        function out = getBinFactor(obj)
            out = obj.binFactor;
        end
        function out = getFilter(obj)
            out = obj.filter;
        end
        function out = getAllFilteredProj(obj)
            out = obj.filteredProj;
        end
        % @param double index, projection number
        function out = getFilteredProj(obj,index)
            out = obj.filteredProj(:,:,index);
        end    
        function out = getInterptype(obj)
            out = obj.interptype;
        end
        function out = getOutputPath(obj)
            out = obj.outputPath;
        end
        function resetDimensions(obj)
            switch obj.axisDirection
                case 'horiz'
                    obj.width = size(obj.projections,1);
                    obj.height = size(obj.projections,2);
                case 'vert'
                    obj.width = size(obj.projections,2);
                    obj.height = size(obj.projections,1);
            end
        end
        function setPath(obj,varargin)
            if nargin==2
                obj.path = varargin{1};
            else
                obj.path = uigetdir;
            end
        end
        %param double R, effective source-detector distance
        function setFilter(obj,filterString)
            obj.filter = filterString;
        end  
        %@param boolean useGPU
        function setUseGPU(obj,useGPU)
            switch useGPU
                case true
                    obj.useGPU = true;
                case false
                    obj.useGPU = false;
                otherwise
                    error('Use GPU must be truee/false');
            end
        end
        %@param string rotDirection, either 'clock' or 'anti'
        function setRotationDirection(obj,rotDirection)
            if strcmp(rotDirection,'clock') || strcmp(rotDirection,'anti')
                obj.rotationDirection = rotDirection;
            else
                error('Rotation direction must be either "clock" or "anti"');
            end
        end
        %@param string axisDirection, either 'horiz' or 'vert'
        function setAxisDirection(obj,axisDirection)
            if strcmp(axisDirection,'horiz') || strcmp(axisDirection,'vert')
                obj.axisDirection = axisDirection;
            else
                error('Rotation direction must be either "horiz" or "vert"');
            end
        end
        %@param double angle, angle of AOR in degrees
        function setAngle(obj,angle)
            obj.motorAxisAngle = angle;
        end
        %@param double angle, total number of angles, either 360 or 180
        function setNAngles(obj,nAngles)
            obj.nAngles = nAngles;
        end
        %@param double[] opticCentre, (x,y) location of optic centre in mm
        function setOpticCentre(obj,opticCentre)
            obj.opticCentre = opticCentre;
        end
        %@param double nProj, set number of projections
        function setNProj(obj,nProj)
            obj.nProj = nProj;
        end
        %@param double pixelSize, size of sensor pixels in mm
        function setPixelSize(obj,pixelSize)
            obj.pxSz = pixelSize;
        end
        %@param double width, pixel distance perpendicular to rotation axis
        %(unbinned)
        function setWidth(obj,width)
            obj.unbinnedWidth = width;
        end
        %@param double width, pixel distance parallel to rotation axis
        %(unbinned)
        function setHeight(obj,height)
            obj.unbinnedHeight = height;
        end
        %@param double binFactor
        function setBinFactor(obj,binFactor)
            obj.binFactor = binFactor;
        end
        %@param double apSz, size of aperture in mm
        function setApertureRadius(obj,apSz)
            obj.apertureRadius = apSz;
        end
        %@param double lambda, wavelength imaging in mm
        function setLambda(obj,lambda)
            obj.lambda = lambda;
        end
        %@param double motorAxisXDisplacement, x displacement of rotation
        %axis from centre of object volume in pixels (unbinned)
        function setMotorAxisXDisplacement(obj,motorAxisXDisplacement)
            obj.motorAxisXDisplacement = motorAxisXDisplacement;
        end
        %@param double motorAxisZDisplacement, z displacement of rotation
        %axis from centre of object volume in pixels (unbinned)
        function setMotorAxisZDisplacement(obj,motorAxisZDisplacement)
            obj.motorAxisZDisplacement = motorAxisZDisplacement;
        end
        % gets offset of focal plane for system centre, in mm
        function out = getFocalPlaneOffset(obj)
            out = obj.focalPlaneOffset;
        end
        
    end
    
    %% abstract methods
    methods (Abstract)
        reconstruct(obj);
    end
    
    %% projection load / manipulation
    methods
        %load projection images from file (assume cropped to centre, if
        %cropped, varargin = nAngles either 180 or 360 (default is 360)
        function loadProjections(obj,varargin)
            obj.projections = [];
            obj.filteredProj = [];
            if nargin<=1   
                obj.path = uigetdir();
            end
            im_dir = dir(fullfile(obj.path,'*.tif'));
            if isempty(im_dir)
                im_dir = dir(fullfile(obj.path,'*.tiff'));
            end
            obj.nProj = length(im_dir);

            for i=1:obj.nProj
                if i==1
                    im = single(imread(strcat(fullfile(obj.path,char(cellstr(im_dir(i).name))))));
                    im_sz = size(im);
                    obj.projections = single(zeros(im_sz(1),im_sz(2),obj.nProj));
                else
                    im = single(imread(strcat(fullfile(obj.path,char(cellstr(im_dir(i).name))))));
                end
                disp(i/obj.nProj*100);
                obj.projections(:,:,i) = single(im);
            end

            if nargin>2
                obj.nAngles = varargin{1};
            else
                obj.nAngles = 360;
            end
            
            figure; for i=1:5:50
                imagesc(obj.projections(:,:,i)); drawnow; pause(0.3); end
            obj.rotationDirection = input('Enter rotation direction (clock/anti): ');
            obj.axisDirection = input('Enter axis of rotation direction (horiz/vert): ');
            
            switch obj.axisDirection
                case 'horiz'
                    obj.realWidth = size(obj.projections,1);
                    obj.realHeight = size(obj.projections,2);
                case 'vert'
                    obj.realWidth = size(obj.projections,2);
                    obj.realHeight = size(obj.projections,1);
                otherwise
                    error('Axis direction must be "horiz" or "vert"');
            end
   
        end
        
        %@param int binFactor, choose to square bin the projections
        %(requires even
        function binProjections(obj,binFactor)
            binnedProj = zeros(floor(size(obj.projections,1)/binFactor),floor(size(obj.projections,2)/binFactor),obj.nProj);
            projOut = zeros(floor(size(obj.projections,1)/binFactor),floor(size(obj.projections,2)/binFactor));
            for i=1:obj.nProj            
                %tic
                currentProj = obj.projections(:,:,i);
                for n=1:binFactor
                    for m = 1:binFactor
                    currentProj = currentProj+circshift(currentProj,[(n-1),(m-1)]);
                    end
                end
                for j = 1:floor(size(obj.projections,1)/binFactor)
                    for k = 1:floor(size(obj.projections,2)/binFactor)
                        projOut(j,k) = currentProj((j-1)*binFactor+1,(k-1)*binFactor+1)/(binFactor*binFactor);
                    end
                end
                %toc
               binnedProj(:,:,i) = projOut;
               disp(i/obj.nProj*100);
            end
            obj.projections = [];
            obj.projections = binnedProj;
            obj.binFactor = binFactor;                   
        end
        
        %display maximum intensity projection of projections
        function getMIP(obj)
            if isempty(obj.MIP)
                obj.MIP = max(obj.projections,[],3);
            end
            figure; imagesc(obj.xPixels*obj.getPixelSize,obj.yPixels*obj.getPixelSize,obj.MIP); 
            title('Projections Maximum Intensity Projection'); xlabel('mm'); axis equal tight;
        end
        
        %sets single projection using simulation functionality
        function setProjection(obj,projection,index)
            obj.projections(:,:,index) = projection;
        end
        
        %sets projeciton parameter of size array with value
        %@param double[] size, size of projection array
        %@param double value, value to initialise array with
        function setAllProjections(obj,value)
            switch obj.getAxisDirection
                case 'horiz'
                    obj.projections = ones(obj.getWidth,obj.getHeight,obj.getNProj).*value;
                case 'vert'
                    obj.projections = ones(obj.getHeight,obj.getWidth,obj.getNProj).*value;
            end
        end
        
        %clears any loaded projections
        function clearProjections(obj)
            obj.projections = [];
            obj.filteredProj = [];
            obj.MIP = [];
        end
        
        % @param double mnidx, minimum slice number index
        % @param double mxidx, maximum slice number index to reconstruct
        % @param boolean displayBoolean, true to display slices
        % @param StepperMotor stepperMotor, provides rotation axis
        % displacement values
        % @param boolean varargin{1}, display boolean, true==new figure
        function reconstructProjections(obj,varargin)
            narginchk(2,8);
            [mnidx,mxidx,stepperMotor,objective,displayBoolean,obStage,tStage] = parse_inputs(varargin{:});
            reconstructionFolderName = 'Reconstructions';
            if ~isdir(fullfile(obj.path,reconstructionFolderName))
                mkdir(obj.path,reconstructionFolderName);
                dlmwrite(fullfile(fullfile(obj.path,reconstructionFolderName),'MaxMinValues.txt'),zeros(obj.getHeight,2),';');
            elseif exist(fullfile(fullfile(obj.path,reconstructionFolderName),'MaxMinValues.txt'))~=2
                dlmwrite(fullfile(fullfile(obj.path,reconstructionFolderName),'MaxMinValues.txt'),zeros(obj.getHeight,2),';');
            end
            obj.outputPath = fullfile(obj.path,reconstructionFolderName);
            if displayBoolean
                figure;
                reconstruct(obj,mnidx,mxidx,stepperMotor,objective,obStage,tStage,true);
            else
                reconstruct(obj,mnidx,mxidx,stepperMotor,objective,obStage,tStage,false);
            end
            
            save(fullfile(obj.path,'objective.mat'),'objective');
            save(fullfile(obj.path,'stepperMotor.mat'),'stepperMotor');
            
            %input parser
            function [mnidx,mxidx,stepperMotor,objective,displayBoolean,obStage,tStage] = parse_inputs(varargin)
                numbers = []; obStage = []; tStage = []; displayBoolean = false;
                objective = []; stepperMotor = [];
                for argNumber=1:nargin
                    arg = varargin{argNumber};
                    type = class(arg);
                    switch type
                        case 'double'
                            numbers = horzcat(numbers,arg);
                        case 'logical'
                            displayBoolean = arg;
                        case 'Objective'
                            objective = arg;
                        case 'PointObject'
                            pointObject = arg;
                        case 'StepperMotor'
                            stepperMotor = arg;
                        case 'ObjectiveStage'
                            obStage = arg;
                        case 'TranslationStage'
                            tStage = arg;
                    end
                end
                if isempty(objective) || isempty(stepperMotor)
                    error('Please assign objective, and stepper motor for reconstruction');
                end
                if length(numbers)==2
                    mnidx = min(numbers); mxidx = max(numbers);
                elseif length(numbers)==1
                    mnidx = numbers; mxidx = numbers;
                else
                    error('Too many number arguments');
                end
            end
                            
        end
        
        function normaliseReconstructions(obj)
            if exist(fullfile(obj.outputPath,'MaxMinValues.txt'))~=2
                error('Please reconstruct projections first');
            else
                maxMinValues = dlmread(fullfile(obj.outputPath,'MaxMinValues.txt'));
                minimumValue = min(maxMinValues(:,1));
                maximumValue = max(maxMinValues(:,2));
                im_dir = dir(fullfile(obj.outputPath,'*.tif'));
                if isempty(im_dir)
                    im_dir = dir(fullfile(obj.outputPath,'*.tiff'));
                end
                for i=1:length(im_dir)
                    im = single(imread(strcat(fullfile(obj.outputPath,char(cellstr(im_dir(i).name))))));
                    im_name = strsplit(im_dir(i).name,'.');
                    index = str2double(im_name{1});
                    im = im./65535*((maxMinValues(index,2)-maxMinValues(index,1))+maxMinValues(index,1));
                    im = uint16((im-minimumValue)./(maximumValue-minimumValue).*65535);
                    imagesc(im); axis equal tight; caxis([0,2^16-1]); drawnow; pause(.3);
                    imwrite(im,strcat(fullfile(obj.getOutputPath,num2str(index,'%05d')),'.tiff'));
                    disp(sprintf('Normalisation Completion Percentage: %.1f%%',i/length(im_dir)*100));
                end 
            end
        end
        
        %@param Objective objective, objective class
        %@param PointObject pointObject, point object class
        %@param PointObject[] pointObject, array of pointObjects
        %@param StepperMotor stepperMotor, stepper motor class provides aor
        %position
        % @param boolean varargin{1}, true if want to display projection
        function simulateProjections(obj,varargin)
            narginchk(1,8);
            [objective,pointObject,stepperMotor,displayBoolean,obStage,tStage] = parse_inputs(varargin{:});
            obj.calculateAF(objective);
            if strcmp(class(obj),'ConeBeamSystem')
                if isempty(obj.getApertureDisplacement)
                    obj.calculateApertureDisplacement(objective);
                end
            end
            if isempty(obj.apertureRadius)
                obj.apertureRadius = 2*objective.getRadiusPP;
            end
            
            obj.clearProjections;
            obj.setAllProjections(0);
            xRange = obj.xPixels.*obj.getPixelSize; % image space real x coords in mm
            yRange = obj.yPixels.*obj.getPixelSize; % image space real y coords in mm
            
            if obj.useGPU==1
                [imageX,imageY] = gpuArray(meshgrid(xRange,yRange));
            else
                [imageX,imageY] = meshgrid(xRange,yRange);
            end

            for i=1:length(obj.theta)
                I = zeros(obj.getHeight,obj.getWidth);
                for currentPointObject = pointObject
                    rotObject = stepperMotor.rotate(currentPointObject,i,obj.theta); % new point object in mm loc after rotation
                    if ~isempty(obStage) 
                        zOffset = obj.getFocalPlaneOffset+obStage.getMotionAtProj(i);
                    else
                        zOffset = obj.getFocalPlaneOffset;
                    end
                    [~,psf,xScale,yScale] = obj.AF.getPSF(obj,objective,rotObject,5,zOffset,'czt');
                    if obj.useGPU==1
                        [psfX,psfY] = gpuArray(meshgrid(xScale,yScale));
                        if ~isempty(tStage)
                            interpPSF = interp2(psfX,psfY,psf,imageX+tStage.getMotionAtProj(i)*objective.getMagnification,imageY);
                        else
                            interpPSF = interp2(psfX,psfY,psf,imageX,imageY);
                        end
                        interp2(psfX,psfY,psf,imageX,imageY);
                        interpPSF(isnan(interpPSF)) = 0;
                        I = I+interpPSF;
                    else
                        [psfX,psfY] = meshgrid(xScale,yScale);
                        if ~isempty(tStage)
                            interpPSF = interp2(psfX,psfY,psf,imageX+tStage.getMotionAtProj(i)*objective.getMagnification,imageY);
                        else
                            interpPSF = interp2(psfX,psfY,psf,imageX,imageY);
                        end
                        interp2(psfX,psfY,psf,imageX,imageY);
                        interpPSF(isnan(interpPSF)) = 0;
                        I = I+interpPSF;
                    end
                end
                obj.setProjection(I,i);
                if displayBoolean
                    if i==1
                        h=figure;
                    end
                    imagesc(h,xRange,yRange,I); xlabel('mm'); title('Binned Image Sensor'); axis equal tight; colorbar; drawnow;
                end
                if rem(i,round(obj.nProj/20))==0
                    disp(sprintf('Simulation Completion Percentage = %.0f%%',i/length(obj.theta)*100));
                end
            end
            
            %input parser
            function [objective,pointObject,stepperMotor,displayBoolean,obStage,tStage] = parse_inputs(varargin)
                obStage = []; tStage = []; displayBoolean = false;
                objective = []; pointObject = []; stepperMotor = [];
                for argNumber=1:nargin
                    arg = varargin{argNumber};
                    type = class(arg);
                    switch type
                        case 'Objective'
                            objective = arg;
                        case 'PointObject'
                            pointObject = arg;
                        case 'StepperMotor'
                            stepperMotor = arg;
                        case 'ObjectiveStage'
                            obStage = arg;
                        case 'TranslationStage'
                            tStage = arg;
                        case 'logical'
                            displayBoolean = arg;
                    end
                end
                if isempty(objective) || isempty(pointObject) || isempty(stepperMotor)
                    error('Please assign objective, object and stepper motor for simulation');
                end
            end
        end

        %@param double index, index of sinogram
        function out = getSinogram(obj,index)
            switch obj.axisDirection
                case 'horiz'
                    out = squeeze(obj.projections(:,index,:));
                case 'vert'
                    out = squeeze(obj.projections(index,:,:));
            end
        end
        
        %@param double index, index of sinogram
        %@param StepperMotor stepperMotor
        %@param Objective objective
        %@param TranslationStage tStage, [] if no stage
        function out = getShiftedSinogram(obj,index,stepperMotor,objective,tStage)
            aorVector = (stepperMotor.xDisplacement(obj.theta)+stepperMotor.getX)/obj.getPixelSize*objective.getMagnification;
            if ~isempty(tStage)
                aorVector = aorVector+tStage.discreteDifference/obj.getPixelSize*objective.getMagnification;
            end
            switch obj.axisDirection
                case 'horiz'
                    out = OPTSystem.shiftSinogram(squeeze(obj.projections(:,index,:)),aorVector);
                case 'vert'
                    out = OPTSystem.shiftSinogram(squeeze(obj.projections(index,:,:)),aorVector);
            end
        end
        
        % returns current object space depth of field in binned image space pixels
        %@param double n, refractive index of immersion fluid
        %@param Objective objective
        function out = DoF(obj,objective)
            DoFmm = objective.getEffDoF(obj.pxSz,obj.lambda,obj.apertureRadius,obj.n);
            out = DoFmm*objective.getMagnification/obj.getPixelSize;
        end
        
        % returns current object space depth of field in mm
        %@param double n, refractive index of immersion fluid
        %@param Objective objective
        function out = DoFinMM(obj,objective)
            out = objective.getEffDoF(obj.pxSz,obj.lambda,obj.apertureRadius,obj.n);
        end
        
        % returns current system object space NA
        %@param Objective objective
        function out = NA(obj,objective)
            out = objective.getEffNA(obj.apertureRadius);
        end
               
        % @param Objective objective
        % @return double out, FWHM spatial resolution of current setup in
        % mm
        function out = resolution(obj,objective)
            out = obj.lambda/(2*objective.getEffNA(obj.apertureRadius));
        end
        
        % setup for full DoF OPT - sets the system aperture radius and
        % motor axis location
        % @param double n, refractive index of immersion fluid
        % @param Objective objective
        % @param StepperMotor stepperMotor
        function setupFullDoFOPT(obj,objective,stepperMotor,n)
            DoF = obj.getWidth*obj.getPixelSize/objective.getMagnification;
            NA = sqrt(obj.lambda*n/DoF+(n*obj.pxSz/objective.getMagnification/2/DoF)^2)+n*obj.pxSz/objective.getMagnification/2/DoF;
            obj.apertureRadius = NA/objective.getNA*objective.getRadiusPP;
            stepperMotor.setZ(0);
            obj.focalPlaneOffset = 0;
        end
        
        % setup for half DoF OPT - sets the system aperture radius and
        % motor axis location
        % @param double n, refractive index of immersion fluid
        % @param Objective objective
        function setupHalfDoFOPT(obj,objective,stepperMotor,n,varargin)
            if nargin==4
                DoF = obj.getWidth*obj.getPixelSize/2/objective.getMagnification;
                NA = sqrt(obj.lambda*n/DoF+(n*obj.pxSz/objective.getMagnification/2/DoF)^2)+n*obj.pxSz/objective.getMagnification/2/DoF;
                obj.apertureRadius = NA/objective.getNA*objective.getRadiusPP;
            elseif nargin==5
                DoF = obj.DoF(objective,n)*obj.getPixelSize/objective.getMagnification;
            end
            obj.focalPlaneOffset = DoF/2;
            stepperMotor.setZ(0);
        end
    
    end
    
    %% Ambiguity Function and OTFS
    methods
       % generates AmbiguityFunction object, and fills value using obj and objective parameters
       % @param Objective objective
       % @param boolean varargin, display boolean, true==display 
       function calculateAF(obj,objective,varargin)
            obj.AF = AmbiguityFunction();
            if nargin==3 && varargin{1}
                obj.AF.generate(obj,objective,true)
            else
                obj.AF.generate(obj,objective);
            end
       end
       
       % calculates object space psf XZ profile to first zeros
       % @oaram Objective objective
       % @param String limit, either 'dof' or 'zeros' - extent of z range
       % either to 2x definition of depth of field or to the first zeros
       % @param PointObject varargin, optional point object to probe system
       % at objects x,y coordinates
       % @return double[][], xzPSF
       function [xzPSF,dispScale,zRange] = xzPSF(obj,objective,limit,varargin)
           if nargin==3
               object = PointObject(0,0,0);
           elseif nargin~=4
               error('Incorrect number of arguments');
           else
               object = PointObject(varargin{1}.getX,varargin{1}.getY,0);
           end
           obj.calculateAF(objective);
           numberBesselZeros = 5;
           switch limit
               case 'zeros'
                   zLimit = 2*obj.getLambda/(objective.getEffNA(obj.getApertureRadius)^2);
               case 'dof'
                   zLimit = objective.getEffDoF(obj.pxSz,obj.lambda,obj.apertureRadius,1);
               otherwise
                   error('Limit must be zeros or dof');
           end
           zRange = linspace(-zLimit,zLimit,256);
           zOffset = obj.getFocalPlaneOffset;
           psfFocus = obj.AF.getPSF(obj,objective,object,numberBesselZeros,zOffset,'czt');
           [~,dispScale,rowIndex] = psfFocus.xProfile;
           dispScale = dispScale/objective.getMagnification; %switch to object-space scale
           for i=1:length(zRange)
               point = PointObject(object.getX,object.getY,zRange(i));
               psfObject = obj.AF.getPSF(obj,objective,point,numberBesselZeros,zOffset,'czt');
               sysType = class(obj);
               if strcmp(sysType,'ConeBeamSystem')
                   if isempty(obj.getApertureDisplacement)
                       obj.calculateApertureDisplacement(objective); 
                   end
                    xzPSF(i,:) = interp1(psfObject.getXScale/objective.getMagnification,psfObject.rowProfile(rowIndex),dispScale);
               elseif strcmp(sysType,'Standard4fSystem')
                   xzPSF(i,:) = psfObject.rowProfile(rowIndex);
               end
               if rem(i,round(length(zRange)/20))==0
                disp(sprintf('xzPSF completion percentage: %.1f%%',i/length(zRange)*100));
               end
           end
           figure; imagesc(dispScale,zRange,xzPSF./max(xzPSF(:))); xlabel('x (mm)'); ylabel('z (mm)'); title('Object space PSF axial profile'); axis square;
           figure; plot(zRange,xzPSF(:,size(xzPSF,2)/2+1)./max(xzPSF(:))); 
       end                      
    end

    %% Protected Methods (only accessed by super and subclasses)
    methods (Access = protected)
        %filters projections (only ram-lak functionality implemented)
        function filterProjections(obj)
            %FILTERING with x,y,z centred on object with u offset on optic axis.            
            switch obj.axisDirection
                case 'horiz'
                    ftDimension = 1;
                    if isempty(obj.filteredProj)
                        obj.filteredProj = single(zeros(obj.getWidth,obj.getHeight,obj.nProj));
                    end
                    currentFilter = obj.generateFilter;
                    padSizeVector = [(size(currentFilter,1)-size(obj.projections,1))/2,0];
                case 'vert'
                    ftDimension = 2;
                    if isempty(obj.filteredProj)
                        obj.filteredProj = single(zeros(obj.getHeight,obj.getWidth,obj.nProj));
                    end
                    currentFilter = obj.generateFilter;
                    padSizeVector = [0,(size(currentFilter,2)-size(obj.projections,2))/2];
                otherwise
                    error('Incorrect axisDirection property');
            end
            
            fillIndexes = size(currentFilter,2)/2-size(obj.projections,2)/2+1:size(currentFilter,2)/2-size(obj.projections,2)/2+obj.getWidth;
            for i=1:obj.nProj
                if rem(i,round(obj.nProj/20))==0 
                    disp(sprintf('Filtration completion percentage: %.1f%%',i/obj.nProj*100));
                end
                p = obj.projections(:,:,i);
                fproj = padarray(p,padSizeVector,'replicate','both');
                fproj = fft(fproj,[],ftDimension);
                fproj = fproj.*currentFilter;
                fproj = real(ifft(fproj,[],ftDimension));
                if obj.useGPU==1
                    if ftDimension==1
                        obj.filteredProj(:,:,i) = gather(fproj(fillIndexes,:));
                    else
                        obj.filteredProj(:,:,i) = gather(fproj(:,fillIndexes));
                    end
                else
                    if ftDimension==1
                        obj.filteredProj(:,:,i) = fproj(fillIndexes,:);
                    else
                        obj.filteredProj(:,:,i) = fproj(:,fillIndexes);
                    end
                end
            end
        end 
    end
    
    %% Secondary Parameters
    methods
        function out = theta(obj)
            switch obj.rotationDirection
                case 'anti'
                    out = linspace(0,obj.nAngles/360*2*pi-obj.nAngles/360*2*pi/obj.nProj,obj.nProj);
                case 'clock'
                    out = linspace(0,-obj.nAngles/360*2*pi+obj.nAngles/360*2*pi/obj.nProj,obj.nProj);
                otherwise
                    error('Please set a rotation direction');
            end
        end       
        function out = xPixels(obj)
            out = (-obj.getWidth/2:obj.getWidth/2-1); % scale of object x in binned pixels
        end        
        function out = yPixels(obj)
            out = (-obj.getHeight/2:obj.getHeight/2-1); % scale of object y in binned pixels
        end       
        function out = zPixels(obj)
            out = (-obj.getWidth/2:obj.getWidth/2-1); % scale of object z in binned pixels
        end
    end
    
    
    %% Private and Static Methods
    methods (Access = private)        
        function filt = generateFilter(obj)
            %generate filter with x,y,z centred on axis of rotation.   
            filt_len = max(64,2^nextpow2(2*obj.getWidth-1));
            switch obj.axisDirection
                case 'horiz'
                    [ramp_kernel] = repmat(OPTSystem.ramp_flat(obj.getWidth),1,obj.getHeight);            
                    ramp_kernel = padarray(ramp_kernel,[(filt_len-obj.getWidth)/2,0],'both');
                    filt = abs(fft(ramp_kernel,[],1));
                case 'vert'
                    [ramp_kernel] = repmat(OPTSystem.ramp_flat(obj.getWidth).',obj.getHeight,1);
                     ramp_kernel = padarray(ramp_kernel,[0,(filt_len-obj.getWidth)/2],'both');
                     %ramp_kernel = padarray(ramp_kernel,[0,filt_len/2-obj.getWidth/2+obj.getMotorAxisXDisplacement],'pre');
                     %ramp_kernel = padarray(ramp_kernel,[0,filt_len/2-obj.getWidth/2-obj.getMotorAxisXDisplacement],'post');
                     filt = abs(fft(ramp_kernel,[],2));
            end
                    
            if (obj.useGPU==1)
                filt = gpuArray(filt);
            end
        end
    end
    
    methods (Static)
        % @param double n, width of ramp filter
        function [h, nn] = ramp_flat(n)
            nn = [-(n/2):(n/2-1)]';
            h = zeros(size(nn),'single');
            h(n/2+1) = 1 / 8;
            odd = mod(nn,2) == 1;
            h(odd) = -0.5 ./ (pi * nn(odd)).^2;
        end
        %@param double[][], sinogram to shift, 2D cross section of
        %projection data perpendicular to rotation axis. Columns are
        %different projection angles
        %param double[], aorVector, shifting for each projection angle
        %param double aorVector, constant shift for all projections
        function out = shiftSinogram(sinogram,aorVector)
            out = zeros(size(sinogram));
              if length(aorVector)==1
                  y = (1:size(sinogram,1))-aorVector;
                  for i=1:size(sinogram,2)
                      out(1:size(sinogram,1),i) = interp1(sinogram(:,i),y);
                  end
              elseif length(aorVector)==size(sinogram,2)
                  for i=1:size(sinogram,2)
                      y = (1:size(sinogram,1))-aorVector(i);
                      out(1:size(sinogram,1),i) = interp1(sinogram(:,i),y);
                  end
              else
                  error('Invalid AoR vector amount');
              end
            out(isnan(out)) = 0;
        end
    end
        
    
end
