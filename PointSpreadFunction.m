classdef PointSpreadFunction < handle
    %Class with basic PSF properties, value, scale and methods to calculate
    %FWHM from gaussian fitting. Method to ramp-filter point spread
    %function
    
    properties (Access=private)
        value %double[][] 2D array of PSF values
        xScale %x-scale in mm
        yScale %y-scale in mm
        FWHM %full-width half maximum spatial resolution in mm
        rampValue %double[][] 2D array of ramp filtered PSF
        
        maxProfile %profile through PSF that maximises resoltuion
        minProfile %profile through PSF that minimises resolution
        maxFWHM % max FWHM through anisotropic PSF
        minFWHM %min FWHM through anisotropic PSF
    end
    
    methods %constructor and get/set
        function obj = PointSpreadFunction(varargin)
            if nargin==3
                obj.value = varargin{1};
                obj.xScale = varargin{2};
                obj.yScale = varargin{3};
            elseif nargin~=0
                error('Send value,xScale and yScale or no arguments');
            end
        end
        % @return double out, fwhm resolution
        function out = getFWHM(obj)
            if isempty(obj.FWHM)
                obj.calculateFWHM;
            end
            out = obj.FWHM;
        end
        % @return double[] out, x-scale in mm
        function out = getXScale(obj)
            out = obj.xScale;
        end
        % @return double[] out, y-scale in mm
        function out = getYScale(obj)
            out = obj.yScale;
        end
        % @return double[][] out, value of psf
        function out = getValue(obj)
            out = obj.value;
        end
        % @param double[][] value, value of psf
        function setValue(obj,value)
            obj.value = value;
        end
        % @param double[] xScale,x-scale in mm
        function setXScale(obj,xScale)
            obj.xScale = xScale;
        end
        % @param double[] yScale, y-scale in mm
        function setYScale(obj,yScale)
            obj.yScale = yScale;
        end
        % @param double[][] rampValue, rampValue filtered psf
        function setRampValue(obj,rampValue)
            obj.rampValue = rampValue;
        end
        % @return double[][] varargout, optional returns value
        function varargout = show(obj)
            figure;
            imagesc(obj.xScale,obj.yScale,obj.value); title('PSF'); xlabel('(mm)'); axis square;
            varargout{1} = obj.value;
        end
        % @return complex[][] varargout, optional returns ramp value
        function varargout = showRamp(obj)
            if ~isempty(obj.rampValue)
                figure;
                imagesc(obj.xScale,obj.yScale,real(obj.rampValue)); title('PSF'); xlabel('(mm)'); axis square;
                varargout{1} = obj.rampValue;
            else
                error('Please calculate ramp value first');
            end
        end
        % @param int rowIndex, gets row profile of rowIndex
        % @return double[] out, psf row profile
        function out = rowProfile(obj,rowIndex)
            out = obj.value(rowIndex,:);
        end
         % @param int colIndex, gets column profile of rowIndex
        % @return double[] out, psf column profile
        function out = colProfile(obj,colIndex)
            out = obj.value(:,colIndex);
        end

        function out = getMinProfile(obj)
            out = obj.minProfile;
        end
        function out = getMaxProfile(obj)
            out = obj.maxProfile;
        end
        function out = getMaxFWHM(obj)
            out = obj.maxFWHM;
        end
        function out = getMinFWHM(obj)
            out = obj.minFWHM;
        end
        
    end
    
    methods
        % @param boolean varargin, display boolean, true==new figure
        function out = calculateFWHM(obj,varargin)
            if ~isempty(obj.value) && ~isempty(obj.xScale) && ~isempty(obj.yScale)
                if nargin==2 && varargin{1}
                    [obj.FWHM,gaussianFit] = PointSpreadFunction.fit2DGaussian(obj.value,obj.xScale,obj.yScale,true);
                else
                    [obj.FWHM,gaussianFit] = PointSpreadFunction.fit2DGaussian(obj.value,obj.xScale,obj.yScale);
                end
                figure; plot(obj.xScale,obj.value(size(obj.value,1)/2+1,:)); hold on; plot(obj.xScale,gaussianFit(size(obj.value,1)/2+1,:)); 
                xlabel('x (mm)'); ylabel('Intensity'); hold off;  title('1D profile through fit');drawnow;
                out = obj.getFWHM;
            else
                error('Please set PSF value and scales before calculating fwhm');
                out = [];
            end
        end
        function rampFilter(obj)
            if ~isempty(obj.value)
                nPx = size(obj.value,2);
                ramp = repmat((abs(-nPx/2:nPx/2-1)*2/nPx),size(obj.value,1),1);
                obj.rampValue = ifftshift(ifft2(ifftshift(fftshift(fft2(fftshift(obj.value))).*ramp)));
            else
                error('Please set PSF value before calculating ramp filter');
            end
        end
        % @return double[] rPSF, returns x-profile of psf
        % @return double[] scale, scale in mm
        % @return int row, row of profile
        function [xPSF,scale,row] = xProfile(obj)
            if ~isempty(obj.value)
                [row,~] = find(obj.value == max(obj.value(:)));
                xPSF = obj.value(row,:);
                scale = obj.xScale;
            else
                error('No PSF value');
            end
        end
        % @return double sum, sum of PSF
        function out = sum(obj)
            out = sum(obj.value(:));
        end
        
        %method to calculate resolution by shifting one of a pair of psfs
        %and looking for a peak prominence of 1/2 maximum
        % @param varargin:
            % @param double angle, angle of line through 2D PSF to find
            % resolution
        function [out,psfLine] = resolution(obj,varargin)
            [r,c] = find(obj.value==max(obj.value(:)));
            if nargin==1
                psfLine = obj.value(r,:);
            else
                angle = varargin{1};
                x = (1:size(obj.value,2))-c; z = (1:size(obj.value,1))-r;
                [x2D,z2D] = meshgrid(x,z);
                xR = x2D.*cos(angle)+z2D.*sin(angle);
                zR = z2D.*cos(angle)-x2D.*sin(angle);
                xQuery = xR(length(x)/2+1,:)-x(1)+1;
                zQuery = zR(length(z)/2+1,:)-z(1)+1;
                psfLine = interp2(obj.value,xQuery,zQuery);
            end
            halfMaximum = max(obj.value(:))/2;
            start_guesses = 0;
            
            model = @fun;
            options = optimset('MaxFunEvals',2000*length(start_guesses),'MaxIter',2000*length(start_guesses));
            estimates = fminsearch(model,start_guesses,options); 
            [~,sepPeaks] = model(estimates);
            out = estimates(1);
            
            
            %figure; plot(obj.xScale,sepPeaks); xlabel('mm'); title(sprintf('Image-Space Resolution by Peak-Seperation = %.4fmm',out)); drawnow;
            
            function [sse,C] = fun(params)
                shift = params(1);
                
                A = interp1(obj.xScale,psfLine,obj.xScale+shift/2);
                B = interp1(obj.xScale,psfLine,obj.xScale-shift/2);
                
                % This method provides true resolution if defined by
                % peak-dip = 0.5 peak height
                C = A+B;
                % This method provides the conventional answer (i.e 1/abbe)
                C = max(A,B);
                
                sse = sum((C(length(psfLine)/2+1)-halfMaximum).^2);
                
                %plot(obj.xScale,C); drawnow;
                
                
            end
        end
        
        
        % @param varargin:
            % @param string 'constrain', this forces the two profiles to be
            % orthogonal
        function [minLine,minLineFWHM,maxLine,maxLineFWHM,angleMin,angleMax] = maxMinProfiles(obj,varargin)
            if nargin>1 && strcmp(varargin{1},'constrain')
                orthoBool = 1;
            else
                orthoBool =0;
            end
            
            halfMax = max(obj.value(:))/2;
            C = contourc(obj.value,[halfMax,halfMax]);
            
            [r,c] = find(obj.value==max(obj.value(:)));
            x = (1:size(obj.value,2))-nanmean(c); z = (1:size(obj.value,1))-nanmean(r);
            [x2D,z2D] = meshgrid(x,z);
            
            
            start_guesses = [0.05,0.05+pi/2];
            model = @fun;
            estimates = fminsearch(model,start_guesses);
            
            [~,line,line90] = model(estimates);
            lineFWHM = PointSpreadFunction.resolutionFWHM(line,obj.xScale);
            line90FWHM = PointSpreadFunction.resolutionFWHM(line90,obj.xScale);
                
            if lineFWHM>line90FWHM
                minLine = line90; minLineFWHM = line90FWHM; angleMin = estimates(2);
                maxLine = line; maxLineFWHM = lineFWHM; angleMax = estimates(1); 
            else
                minLine = line; minLineFWHM = lineFWHM; angleMin = estimates(1);
                maxLine = line90; maxLineFWHM = line90FWHM; angleMax = estimates(2);
            end
            obj.maxProfile = maxLine; obj.minProfile = minLine;
            obj.maxFWHM = maxLineFWHM; obj.minFWHM = minLineFWHM;
            
            function [sse,line,line90] = fun(params)
                theta = params(1);
                if orthoBool==0
                    theta90 = params(2);
                else
                    theta90 = theta+pi/2;
                end
                
                xThetaQuery = x.*cos(theta); zThetaQuery = z.*sin(theta);
                xTheta90Query = x.*cos(theta90); zTheta90Query = z.*sin(theta90);
                
               % imagesc(x,z,obj.value); axis square; hold on; plot(xThetaQuery,zThetaQuery,'r'); plot(xTheta90Query,zTheta90Query,'g'); hold off; drawnow;
                
                line = interp2(x2D,z2D,obj.value,xThetaQuery,zThetaQuery);
                line90 = interp2(x2D,z2D,obj.value,xTheta90Query,zTheta90Query);

               % sse = 1./sum((line-line90).^2);
                
                sse = 1./nansum((nanstd(line)-nanstd(line90)).^2);
                
            end
        end
        
        % gets row and column profiles around average maximum
        function rowColProfiles(obj)
            [r,c] = find(obj.value==max(obj.value(:)));
            
            line = obj.value(:,nanmean(c)).'; line90 = obj.value(nanmean(r),:);
            lineFWHM = PointSpreadFunction.resolutionFWHM(line,obj.xScale);
            line90FWHM = PointSpreadFunction.resolutionFWHM(line90,obj.xScale);
                
            if lineFWHM>line90FWHM
                minLine = line90; minLineFWHM = line90FWHM;
                maxLine = line; maxLineFWHM = lineFWHM;
            else
                minLine = line; minLineFWHM = lineFWHM;
                maxLine = line90; maxLineFWHM = line90FWHM;
            end
            obj.maxProfile = maxLine; obj.minProfile = minLine;
            obj.maxFWHM = maxLineFWHM; obj.minFWHM = minLineFWHM;
        end
        
        % finds resolution using a contour method
        function contourFWHM(obj)
            interpFactor = 1;
            interpValues = (1:interpFactor:size(obj.value,1));
            overSampleScale = interp1(obj.xScale,interpValues);
            [xSample,ySample] = meshgrid(interpValues);
            overSampledPSF = interp2(obj.value,xSample,ySample);
            
            halfMax = (nanmax(obj.value(:))-nanmedian(obj.value(:)))/2+nanmedian(obj.value(:)); % defined as value at half height between maximum and median value (for noise purposes)
            C = contourc(overSampleScale,overSampleScale,overSampledPSF,[halfMax,halfMax]);
            
            r = sqrt(C(1,2:end).^2+C(2,2:end).^2);
            theta = atan2(C(1,2:end),C(2,2:end));
            
            thetaPi = theta(theta>0); rPi = r(theta>0);
            thetaQuery = thetaPi-pi;
            
            for i=1:length(thetaPi)
                df = abs(theta-thetaQuery(i));
                [~,lc] = min(df);
                FWHM(i) = r(lc)+rPi(i); angle(i) = thetaPi(i);
            end
            
            [maxFWHM,mxLc] = max(FWHM); [minFWHM,mnLc] = min(FWHM);
            angleMax = angle(mxLc); angleMin = angle(mnLc);
            
            [x2D,y2D] = meshgrid(obj.xScale);
            xThetaQueryMax = obj.xScale.*cos(angleMax); zThetaQueryMax = obj.xScale.*sin(angleMax);
            xThetaQueryMin = obj.xScale.*cos(angleMin); zThetaQueryMin = obj.xScale.*sin(angleMin);
            
            minLine = interp2(x2D,y2D,obj.value,xThetaQueryMin,zThetaQueryMin);
            maxLine = interp2(x2D,y2D,obj.value,xThetaQueryMax,zThetaQueryMax);

            obj.maxProfile = maxLine; obj.minProfile = minLine;
            obj.maxFWHM = maxFWHM; obj.minFWHM = minFWHM;
            
            figure; subplot(1,2,1); contour(overSampleScale,overSampleScale,overSampledPSF,[halfMax,halfMax]); axis square;
            subplot(1,2,2); plot(obj.xScale,obj.minProfile); hold on; plot(obj.xScale,obj.maxProfile); hold off; xlabel ('mm'); ylabel('Relative Intensity'); axis square; 
            legend(sprintf('FWHM = %0.2fµm',minFWHM*10^3),sprintf('FWHM = %0.2fµm',maxFWHM*10^3));
            drawnow;
        end
    end

    methods (Static)
        % @param double[][] xydata, 2D image data to fit single 2D gaussian
        % @param double[] xScale, x-scale of xydata in mm
        % @param double[] yScale, y-scale of xydata in mm
        % @param boolean varargin, display boolean, true==new figure
        % @return double FWHM, full-width half maximum spatial resolution in mm
        % @return double[][] gaussianFit
        function [averageFWHM,gaussianFit,eccentricity,minFWHM,maxFWHM,angle] = fit2DGaussian(xydata,xScale,yScale,varargin)
            % T.Watson based on fminsearch model
            [xdata,ydata] = meshgrid(xScale,yScale);
            [row,col] = find(xydata==max(xydata(:)));

            row = round(mean(row));
            col = round(mean(col));
            
            amplitudeGuess = xydata(row,col);
            sigmaGuess = (max(xScale)-min(xScale))/100;
            x0guess = xdata(row,col);
            y0guess = ydata(row,col);
            offsetGuess = std(xydata(:));

            start_guesses = [amplitudeGuess,0,sigmaGuess,sigmaGuess,x0guess,y0guess,offsetGuess];
            
            if nargin==4 && varargin{1}
                drawBoolean = true;
                figure;
            else
                drawBoolean = false;
            end
            
            model = @fun;
            options = optimset('MaxFunEvals',2000*length(start_guesses),'MaxIter',2000*length(start_guesses));
            estimates = fminsearch(model,start_guesses,options);
            [~,gaussianFit] = model(estimates);
            
            averageFWHM = (estimates(3)+estimates(4))/2*2*sqrt(2*log(2));
            minFWHM = min(estimates(3),estimates(4))*2*sqrt(2*log(2));
            maxFWHM = max(estimates(3),estimates(4))*2*sqrt(2*log(2));
            
            eccentricity = sqrt(1-(min(estimates(3),estimates(4))/max(estimates(3),estimates(4)))^2);
            angle = estimates(2);

            function [sse,fit] = fun(params)
                A = params(1);
                theta = params(2);
                sigx = params(3);
                sigy = params(4);
                x0 = params(5);
                y0 = params(6);
                B = params(7);

                a = cos(theta)^2/2/sigx^2+sin(theta)^2/2/sigy^2;
                b = -sin(2*theta)/4/sigx^2+sin(2*theta)/4/sigy^2;
                c = sin(theta)^2/2/sigx^2+cos(theta)^2/2/sigy^2;

                fit = A.*exp(-(a.*(xdata-x0).^2+2*b.*(xdata-x0).*(ydata-y0)+c.*(ydata-y0).^2))+B;
                if drawBoolean
                    subplot(1,3,1)
                    imagesc(xScale,yScale,xydata); axis square;
                    subplot(1,3,2)
                    imagesc(xScale,yScale,fit); axis square;
                    subplot(1,3,3)
                    imagesc(xScale,yScale,fit-xydata); axis square;
                    drawnow;
                end      
                sse = sum(sum((fit-xydata).^2));
            end
        end
        
        % function to calculate FWHM resolution by shifting copies of line
        % @param double[] lineProfile, profile to find FWHM
        % @param double[] scaleProfile, scale of profile in mm
        function out = resolutionFWHM(lineProfile,scaleProfile)
            halfMaximum = max(lineProfile)/2;
            model = @fun;
            start_guesses = 0;
            options = optimset('MaxFunEvals',2000*length(start_guesses),'MaxIter',2000*length(start_guesses));
            estimates = fminsearch(model,start_guesses,options);
            out = estimates(1);
            
            function [sse,C] = fun(params)
                shift = params(1);
                
                A = interp1(scaleProfile,lineProfile,scaleProfile+shift/2);
                B = interp1(scaleProfile,lineProfile,scaleProfile-shift/2);
                
                % This method provides true resolution if defined by
                % peak-dip = 0.5 peak height
                C = A+B;
                % This method provides the conventional answer (i.e 1/abbe)
                C = max(A,B);
                
                sse = sum((C(length(lineProfile)/2+1)-halfMaximum).^2);
                
                %plot(obj.xScale,C); drawnow;
            end
        end
        
    end
    
end

