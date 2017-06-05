classdef ConeBeamSystem < OPTSystem
    %Simulation of point objects in cone-beam OPT system
    
    properties (Access = private)
        apertureDisplacement %axial displacement of aperture from pupil plane mm
        R %effective source-detector distance (pixels) (unbinned raw value)
    end
    
    methods %constructor
        function obj = ConeBeamSystem(varargin)
            obj = obj@OPTSystem();
        end
    end
    
    %% get/set methods
    methods      
        function out = getApertureDisplacement(obj)
            out = obj.apertureDisplacement;
        end
        function out = getR(obj)
            out = obj.R/obj.getBinFactor;
        end
        %@param double R, effective source-detector distance in pixels
        function setR(obj,R)
            obj.R = R;
        end
        %@param double apDisp, axial aperture displacement away from pupil
        %plane in mm
        function setApertureDisplacement(obj,apDisp)
            obj.apertureDisplacement = apDisp;
        end 
        %@param Objective objective, objective class provides focal length
        %and magnfication information
        function calculateApertureDisplacement(obj,objective)
           obj.setApertureDisplacement(objective.getF*objective.getF/(obj.getR.*obj.getPixelSize/objective.getMagnification));
        end         
    end

    methods        
        % @param string outputPath, path to save reconstructions
        % @param double mnidx, minimum index of slices to reconstruct
        % @param double mxidx, maximum index of slices to reconstruct
        % @param boolean displayBoolean, true if want to display
        % @param StepperMotor stepperMotor, class with rotation axis prop
        % @param ObjectiveStage obStage, optional objective stage for focal
        % plane tracking, enter [] if not used
        % @param TranslationStage tStage, optional xy translation stage,
        % enter [] is not used
        function reconstruct(obj,mnidx,mxidx,stepperMotor,objective,~,tStage,displayBoolean)
            if isempty(obj.getAllFilteredProj)
                obj.filterProjections();
            end
            %common parameters to send to reconstruction function
            [xx,zz] = meshgrid(obj.xPixels,obj.zPixels);
            t = obj.theta;
            if obj.getUseGPU == 1
                xxS = gpuArray(xx-stepperMotor.getX/obj.getPixelSize*objective.getMagnification);
                zz = gpuArray(zz);
            else
                xxS = xx-stepperMotor.getX/obj.getPixelSize*objective.getMagnification;
            end
            maxMinValues = dlmread(fullfile(obj.getOutputPath,'MaxMinValues.txt'));
            
            if ~isempty(tStage)
                tStageDiff = tStage.discreteDifference/obj.getPixelSize*objective.getMagnification;
            else
                tStageDiff = zeros(1,obj.getNProj);
            end
            
            for index=mnidx:mxidx
                op = obj.getOpticCentrePixels(objective);        
                D = obj.getR;
                for i = 1:obj.getNProj
                    motorOffset = stepperMotor.currentX(i)/obj.getPixelSize*objective.getMagnification;
                    if obj.getUseGPU==1
                        projI = gpuArray(obj.getFilteredProj(i));
                        if i==1
                            slice = gpuArray(zeros(obj.getWidth,obj.getWidth));
                        end
                    else
                        if i==1
                            slice = zeros(obj.getWidth,obj.getWidth);
                        end
                        projI = (obj.getFilteredProj(i));
                    end

                   u = D.*(xxS*cos(t(i))+zz*sin(t(i))-op(1)+motorOffset)...
                       ./(xxS*sin(t(i))-zz*cos(t(i))+D)+op(1)-motorOffset-tStageDiff(i);

                   U = (xxS*sin(t(i))-zz*cos(t(i))+D)./D;
                   v = (1./U.*(index-obj.getHeight/2-op(2)))+op(2);                       

                    u2 = u-xxS(1,1)+1;
                    v2 = v-zz(1,1)+1;

                    switch obj.getAxisDirection
                        case 'horiz'
                            z = interp2(projI,v2,u2,obj.getInterptype);
                        case 'vert'
                            z = interp2(projI,u2,v2,obj.getInterptype);
                    end
                    plane = z.*D./U.^2;

                    plane(isnan(plane))=0;
                    slice = slice + plane;
                    if rem(i,round(obj.getNProj/20))==0
                        disp(sprintf('Reconstruction Completion Percentage: %.1f%%',(obj.getNProj*(index-mnidx)+i)/(obj.getNProj*(mxidx-mnidx+1))*100));
                    end
                end
                slice(isnan(slice)) = 0;
                maxMinValues(index,1:3) = [min(slice(:)),max(slice(:)),index];
                writeSlice = gather(uint16((slice-min(slice(:)))./(max(slice(:))-min(slice(:))).*65535));
                imwrite(writeSlice,strcat(fullfile(obj.getOutputPath,num2str(index,'%05d')),'.tiff'));
                if displayBoolean==1
                    imagesc(slice); axis equal tight; colorbar(); drawnow;
                end
            end
            dlmwrite(fullfile(obj.getOutputPath,'MaxMinValues.txt'),maxMinValues,';');
        end
        
        % calculates magnifcation at z location
        % @param Objective objective
        % @param double z, query distance away from focal plane in mm
        % @param double beta, projection angle in radians
        function out = magnifcationAtDepth(obj,objective,z)
            if isempty(obj.getApertureDisplacement)
                obj.calculateApertureDisplacement(objective);
            end
            out = objective.getMagnification/(1+obj.apertureDisplacement*z/objective.getF^2);
        end
        
        % calculates magnification profil
       % @oaram Objective objective
       % @param String limit, either 'dof' or 'zeros' - extent of z range
       % either to 2x definition of depth of field or to the first zeros or
       % to a custom double numeric limit
       % @param figureHandle varargin{1}, optional figure handle
       % @param boolean varargin{2}, optional draw boolean
       % @return double[] magRatio, magnification ratio
       % @return double[] zRange, z-range of plot
       % @return double zLimit, z-depth limit of plot
       % @return figure figureHandle, handle of figure
        function [magRatio,zRange,zLimit,figureHandle] = magRatioProfile(obj,objective,limit,varargin)
            [figureHandle,displayBoolean] = parse_inputs(varargin{:});
            switch limit
               case 'zeros'
                   zLimit = 2*obj.getLambda/(objective.getEffNA(obj.getApertureRadius)^2);
               case 'dof'
                   zLimit = objective.getEffDoF(obj.getPixelSize/obj.getBinFactor,obj.getLambda,obj.getApertureRadius,1);
               otherwise
                   if isnumeric(limit)
                       zLimit = limit;
                   end
            end
            zRange = linspace(-zLimit,zLimit,256);
            magRatio = 1./(1+obj.apertureDisplacement.*zRange./objective.getF^2);
            if displayBoolean
                if isempty(figureHandle);
                    figureHandle = figure;
                    ax=gca(figureHandle); 
                else
                    ax=gca(figureHandle); hold on;
                end 
                plot(ax,zRange,magRatio); xlabel('z (mm)'); ylabel('Magnification Ratio');  hold off;
                title(sprintf('Objective Magnification: %.0fx, Aperture Radius=%.2fmm',objective.getMagnification,obj.getApertureRadius));
                axis tight square;
            end
            
            %input parser
            function [figureHandle,displayBoolean] = parse_inputs(varargin)
                figureHandle = []; displayBoolean = false;
                for arg = varargin{:}
                    switch class(arg)
                        case 'logical'
                            displayBoolean = arg;
                        case 'matlab.ui.Figure'
                            figureHandle = arg;
                    end
                end
            end
        end
        
        
    end
    
end

