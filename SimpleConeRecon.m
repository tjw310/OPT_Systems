classdef SimpleConeRecon < handle
    %cone reconstruction class
    
    properties (Access = private)
        %Reconstruction parameters
        useGPU = 0; %boolean: true -> use GPU
        interptype = 'linear'; %interpolateion type, set linear default
        path %path of raw projections
        binFactor = 1; %value of bin factor
        
        %Experimental parameters
        motorAxisDisplacement = 0; %displacement of the motor axis from the centre of FOV in pixels (unbinned raw value)
        motorAxisAngle = 0; %angle of axis of rotation relative to the y axis, default 0,in degrees
        nProj %number of projections
        nAngles %number of angles (either 180 or 360)
        R %effective source-detector distance (pixels) (unbinned raw value)
        opticCentre = [0,0] %(x,y) location of optic centre in pixels (unbinned raw value)
        rotationDirection  %defines rotation direction, either 'clock' or 'anti'
        axisDirection  %define rotation axis direction, either 'horiz' or 'vert'
        pxSz = 6.5e-3 %size of sensor pixels in mm (6.5e-3 by default)
        
        %Properties filled in the reconstruction process
        realWidth %number of pixels in direction perpendicular to the rotation axis (not binned)
        realHeight %number of pixels in direction parallel to the rotation axis (not binned)
        filter = 'ram-lak'; %filter used in reconstruction (Ram-Lak default)
        projections % raw projection images
        filteredProj = [] % filtered projections
        MIP = [] %maximum intensity projection through all projection angles
        
    end
    
    methods
        %constructor
        function obj = SimpleConeRecon(varargin)
            obj.path = uigetdir();
        end

        %reconstruct per slice and save to path
        function slice = reconPS(obj,mnidx,mxidx,varargin) 
            if isempty(obj.filteredProj)
                obj.filteredProj = obj.filterProjections(obj.projections);
            end
            %common parameters to send to reconstruction function
            [xx,zz] = meshgrid(obj.x,obj.z);
            figure;
            t = obj.theta;
            if obj.useGPU == 1
                xxS = gpuArray(xx-obj.getMotorAxisDisplacement);
                zz = gpuArray(zz);
            else
                xxS = xx-obj.getMotorAxisDisplacement;
            end
            
            for i=mnidx:mxidx
                tic
                slice = obj.CTbackprojectionSlice(i,xxS,zz,t);
                toc
                slice = gather(real(slice));
                slice = (slice+300).*10;
                %slice(isnan(slice))=0;
                %slice(slice<0)=0;
                if i==mnidx; figure; end; imagesc(slice);
                title(num2str(i)); axis square; colorbar;
                drawnow;
                slice = uint16(slice);
                if i==mnidx
                    if nargin>3
                        name = varargin{1};
      
                    else
                        name = 'reconSlices';
                    end
                    mkdir(obj.path,name);
                    p = fullfile(obj.path,name);
                end
                imwrite(slice,strcat(fullfile(p,num2str(i,'%05d')),'.tiff'));
            end
        end
               
        %reconstruct from iradon
        function recons = reconiradon(obj,mnidx,mxidx)
            figure;
            for i=mnidx:mxidx
                sinogram = squeeze(obj.projections(:,i,:));
                sinogram = gpuArray(obj.shiftSinogram(sinogram));
                recons(:,:,i) = gather(iradon(sinogram,obj.theta,'ram-lak',1,size(sinogram,1)));
                disp(i);
                imagesc(recons(:,:,i));
            end
%             figure;
%             imagesc(squeeze(max(recons,[],2)));
%             figure;
%             imagesc(squeeze(sum(recons,2)));
        end
        
        %% get/ set Methods          
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
            out = obj.realWidth/obj.binFactor;
        end        
        function out = getHeight(obj)
            out = obj.realHeight/obj.binFactor;
        end
        function out = getNAngles(obj)
            out = obj.nAngles;
        end
        function out = getNProj(obj)
            out = obj.nProj;
        end
        function out = getR(obj)
            out = obj.R/obj.binFactor;
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
        function setR(obj,R)
            obj.R = R;
        end      
        %@param string filterString, name of filter type, eg 'ram-lak'
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
            obj.realWidth = width;
        end
        %@param double width, pixel distance parallel to rotation axis
        function setHeight(obj,height)
            obj.realHeight = height;
        end
        %@param double binFactor
        function setBinFactor(obj,binFactor)
            obj.binFactor = binFactor;
        end
                      
        %% projection load / manipulation
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
                [X,Z]= meshgrid(obj.x,obj.z);           
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
        function filterProjections(obj)
            %FILTERING with x,y,z centred on object with u offset on optic axis.            
            switch obj.axisDirection
                case 'horiz'
                    ftDimension = 1;
                    if isempty(obj.filteredProj)
                        obj.filteredProj = single(zeros(obj.width,obj.height,obj.nProj));
                    end
                    currentFilter = obj.generateFilter;
                    padSizeVector = [(size(currentFilter,1)-size(obj.projections,1))/2,0];
                case 'vert'
                    ftDimension = 2;
                    if isempty(obj.filteredProj)
                        obj.filteredProj = single(zeros(obj.height,obj.width,obj.nProj));
                    end
                    currentFilter = obj.generateFilter;
                    padSizeVector = [0,(size(currentFilter,2)-size(obj.projections,2))/2];
                otherwise
                    error('Incorrect axisDirection property');
            end
            
            fillIndexes = size(currentFilter,2)/2-size(obj.projections,2)/2+1:size(currentFilter,2)/2-size(obj.projections,2)/2+obj.width;
            for i=1:obj.nProj
            %for i=1
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
        %simulates projections
        %@param Objective objective, objective class defining mag and NA
        %@param PointObject pointObject, point object class defining
        %location of point object
        %@param ConeBeamSimulation coneSimulation, cone beam simulation
        %class with parameters such as aperture size, displacement and
        %lambda
        function simulateProjections(obj,objective,pointObject,coneSimulation)
            coneSimulation.simulate(obj,objective,pointObject);
        end
            

        function out = x(obj)
            out = (-obj.getWidth/2:obj.getWidth/2-1); % scale of object x in pixels
        end        
        function out = y(obj)
            out = (-obj.getHeight/2:obj.getHeight/2-1); % scale of object y in pixels
        end       
        function out = z(obj)
            out = (-obj.getWidth/2:obj.getWidth/2-1); % scale of object z in pixels
        end
        function out = theta(obj)
            switch obj.rotationDirection
                case 'anti'
                    out = linspace(0,obj.nAngles/360*2*pi-obj.nAngles/360*2*pi/obj.nProj,obj.nProj);
                case 'clock'
                    out = linspace(0,-obj.nAngles/360*2*pi+obj.nAngles/360*2*pi/obj.nProj,obj.nProj);
            end
        end
        
        %@param Objective objective, objective class
        %@param PointObject pointObject, point object class
        function plotTracesOnMIP(obj,objective,pointObject)
            obj.getMIP;
            coords = pointObject.getTrace(obj,objective);
            hold on; plot(coords(1,:),coords(2,:),'r'); hold off;
        end
    end
    
    methods (Access = private)        
        function filt = generateFilter(obj)
            %generate filter with x,y,z centred on axis of rotation.   
            filt_len = max(64,2^nextpow2(2*obj.width-1));
            switch obj.axisDirection
                case 'horiz'
                    [ramp_kernel] = repmat(simpleConeRecon.ramp_flat(obj.width),1,obj.height);            
                    ramp_kernel = padarray(ramp_kernel,[(filt_len-obj.width)/2,0],'both');
                    filt = abs(fft(ramp_kernel,[],1));
                case 'vert'
                    [ramp_kernel] = repmat(simpleConeRecon.ramp_flat(obj.width).',obj.height,1);
                     ramp_kernel = padarray(ramp_kernel,[0,(filt_len-obj.width)/2],'both');
                     filt = abs(fft(ramp_kernel,[],2));
            end
                    
            if (obj.useGPU==1)
                filt = gpuArray(filt);
            end
        end
  
        function slice = CTbackprojectionSlice(obj,sliceNo,xxS,zz,t)            
            op = obj.getOpticCentre;        
            motorOffset = obj.getMotorAxisDisplacement;
            D = obj.getR;
            
            for i = 1:obj.nProj
                if obj.useGPU==1
                    slice = gpuArray(zeros(obj.getWidth,obj.getWidth));
                    projI = gpuArray(obj.filteredProj(:,:,i));
                else
                    slice = zeros(obj.getWidth,obj.getWidth);
                    projI = (obj.filteredProj(:,:,i));
                end
                
                switch obj.rotationDirection
                    case 'anti'
                        beta = (2*pi-t(i));
                    case 'clock'
                        beta = t(i);
                    otherwise
                        error('Rotation direction must be either "clock" or "anti"');
                end

               u = D.*(xxS*cos(beta)+zz*sin(beta)-op(1)+motorOffset)...
                   ./(xxS*sin(beta)-zz*cos(beta)+D)+op(1)-motorOffset;
               
               U = (xxS*sin(beta)-zz*cos(beta)+D)./D;
               
               v = (1./U.*(-obj.width/2-op(2)+sliceNo))+op(2);                       
                              
                u2 = u-xxS(1,1)+1;
                v2 = v-zz(1,1)+1;
                subplot(2,2,1); imagesc(u2); colorbar();
                subplot(2,2,2); imagesc(v2); colorbar();
                
                switch obj.axisDirection
                    case 'horiz'
                        z = interp2(projI,v2,u2,obj.interptype);
                    case 'vert'
                        z = interp2(projI,u2,v2,obj.interptype);
                end
                plane = z.*D./U.^2;
                subplot(2,2,3); imagesc(projI);

                plane(isnan(plane))=0;
                slice = slice + plane;
                subplot(2,2,4); imagesc(plane); drawnow;
            end
        end
               


    end

    methods (Static)      
        function [h, nn] = ramp_flat(n)
            nn = [-(n/2):(n/2-1)]';
            h = zeros(size(nn),'single');
            h(n/2+1) = 1 / 8;
            odd = mod(nn,2) == 1;
            h(odd) = -0.5 ./ (pi * nn(odd)).^2;
        end

        function [filt] = Filter(filter, kernel, order, d)

        f_kernel = abs(fft(kernel))*2;
        filt = f_kernel(1:order/2+1)';
        w = 2*pi*(0:size(filt,2)-1)/order;   % frequency axis up to Nyquist 

        switch lower(filter)
            case 'ram-lak'
                % Do nothing
            case 'shepp-logan'
                % be careful not to divide by 0:
                filt(2:end) = filt(2:end) .* (sin(w(2:end)/(2*d))./(w(2:end)/(2*d)));
            case 'cosine'
                filt(2:end) = filt(2:end) .* cos(w(2:end)/(2*d));
            case 'hamming'  
                filt(2:end) = filt(2:end) .* (.54 + .46 * cos(w(2:end)/d));
            case 'hann'
                filt(2:end) = filt(2:end) .*(1+cos(w(2:end)./d)) / 2;
            otherwise
                filter
                error('Invalid filter selected.');
        end

        filt(w>pi*d) = 0;                      % Crop the frequency response
        filt = [filt , filt(end-1:-1:2)];    % Symmetry of the filter
        return
        end
        
    end
end

