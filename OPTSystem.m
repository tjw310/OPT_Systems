classdef OPTSystem < handle
    %Class defining an optical projection system and its properties
    
    properties (Access = private)
        % Experimental Parameters
        motorAxisDisplacement = 0; %displacement of the motor axis from the centre of FOV in pixels (unbinned raw value)
        motorAxisAngle = 0; %angle of axis of rotation relative to the y axis, default 0,in degrees
        nProj %number of projections
        nAngles %number of angles (either 180 or 360)
        opticCentre = [0,0] %(x,y) location of optic centre in pixels (unbinned raw value)
        rotationDirection  %defines rotation direction, either 'clock' or 'anti'
        axisDirection  %define rotation axis direction, either 'horiz' or 'vert'
        pxSz = 6.5e-3 %size of sensor pixels in mm (6.5e-3 by default)
        lambda %wavelength in mm
        
        % Large Data
        projections %raw projection information
        filteredProj = [] % filtered projections
        MIP = [] %maximum intensity projection through all projection angles
        
        % Computational Parameteres
        filter = 'ram-lak'; %filter used in reconstruction (Ram-Lak default)
        useGPU = 0; %boolean: true -> use GPU
        interptype = 'linear'; %interpolateion type, set linear default
        path %path of raw projections
        binFactor = 1; %value of bin factor
        unbinnedWidth %number of pixels in direction perpendicular to the rotation axis (not binned)
        unbinnedHeight %number of pixels in direction parallel to the rotation axis (not binned)
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
        function out = getOpticCentre(obj)
            out = obj.opticCentre/obj.binFactor;
        end
        function out = getPixelSize(obj)
            out = obj.pxSz*obj.binFactor;
        end
        function out = getMotorAxisDisplacement(obj)
            out = obj.motorAxisDisplacement/obj.binFactor;
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
        %@param double[] opticCentre, (x,y) location of optic centre
        function setOpticCentre(obj,x,y)
            obj.opticCentre = [x,y];
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
        function setWidth(obj,width)
            obj.unbinnedWidth = width;
        end
        %@param double width, pixel distance parallel to rotation axis
        function setHeight(obj,height)
            obj.unbinnedHeight = height;
        end
        %@param double binFactor
        function setBinFactor(obj,binFactor)
            obj.binFactor = binFactor;
        end
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
        function rotProjections(obj)
            angle = obj.motorAxisAngle;
            if angle~=0
                [X,Z]= meshgrid(obj.xPixels,obj.zPixels);           
                X2 = X*cos(angle/180*pi)-Z*sin(angle/180*pi)-X(1,1);
                Z2 = Z*cos(angle/180*pi)+X*sin(angle/180*pi)-Z(1,1);
                obj.opticCentre(1) = obj.opticCentre(1)*cos(angle/180*pi)-obj.opticCentre(2)*sin(angle/180*pi);
                obj.opticCentre(2) = obj.opticCentre(2)*cos(angle/180*pi)+obj.opticCentre(1)*sin(angle/180*pi);

                for i=1:obj.nProj
                    disp(i/obj.nProj*100)
                    single_rot_proj = single(interp2(obj.projections(:,:,i).',X2,Z2,obj.interptype));
                    obj.projections(:,:,i) = single_rot_proj.';  
                end  
                obj.motorAxisAngle = 0;
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
            figure; imagesc(obj.MIP); axis equal tight;
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
                disp(i/obj.nProj*100);
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
                imagesc(obj.filteredProj(:,:,i)); drawnow;
            end
        end 
        %reconstruct using iradon
        % @param double mnidx, minimum slice number index
        % @param double mxidx, maximum slice number index to reconstruct
        % @param boolean showBoolean, true to display slices
        function reconstructProjections(obj,mnidx,mxidx,showBoolean)
            reconstructionFolderName = 'iradon_Reconstructions';
            if exist(reconstructionFolderNamge,obj.path)~=7
                mkdir(obj.path,reconstructionFolderName);
            end
            outputPath = fullfile(obj.path,name);
            if showBoolean==1
                figure;
            end
            for index=mnidx:mxidx
                if obj.useGPU==1
                    sinogram = gpuArray(obj.getSinogram(index));
                    reconstrucedSlice = gather(iradon(sinogram,obj.theta,obj.filter,1,size(sinogram,1)));
                else
                    sinogram = obj.getSinogram(index);
                    reconstrucedSlice = iradon(sinogram,obj.theta,obj.filter,1,size(sinogram,1));
                end
                imwrite(reconstrucedSlice,strcat(fullfile(outputPath,num2str(index,'%05d')),'.tiff'));
                if showBoolean==1
                    imagesc(reconstrucedSlice); drawnow;
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
                     filt = abs(fft(ramp_kernel,[],2));
            end
                    
            if (obj.useGPU==1)
                filt = gpuArray(filt);
            end
        end
        
        %@param double index, index of sinogram
        %@param varargin
        %           - double[] aorVector, length obj.nProj of a variable
        %           motorAxisDisplacement, position of rotation axis
        %           - double aorVector, constant motorAxisDisplacement
        %           - [], uses existing obj.motorAxisDisplacement (default 0)
        function out = getSinogram(obj,index,varargin)
            switch varargin
                case length(varargin{1})==obj.nProj
                     aorVector = varargin{1};
                case []
                    aorVector = obj.motorAxisDisplacement;
                case length(varargin{1})==1 && isnumeric(varargin{1})
                    aorVector = varargin{1};
                    obj.setMotorAxisDisplacement(varargin{1});
                otherwise
                    error('Incompatiable sinogram shift amount, enter constant or vector of length no.Projections');
            end
            
            switch obj.axisDirection
                case 'horiz'
                    out = OPTSystem.shiftSinogram(squeeze(obj.projections(index,:,:)),aorVector);
                case 'vert'
                    out = OPTSystem.shiftSinogram(squeeze(obj.projections(:,index,:)),aorVector);
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
              switch aorVector
                  case length(aorVector)==1
                      y = 1:size(sinogram,1)-aorVector;
                      for i=1:size(sinogram,2)
                          out(1:size(sinogram,1),i) = interp1(sinogram(:,i),y);
                      end
                  case length(aorVector)==size(sinogram,2)
                      for i=1:size(sinogram,2)
                          y = 1:size(sinogram,1)-aorVector(i);
                          out(1:size(sinogram,1),i) = interp1(sinogram(:,i),y);
                      end
                  otherwise
                      error('Invalid AoR vector amount');
              end
        end
    end
        
    
end

