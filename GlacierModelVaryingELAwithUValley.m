% Glacier Mass Balance and Flow Model: Oscillating ELA with Concave
% Topography

% written by Emily Fairfax
% February 20th, 2016
% with plotting guidance from Megan Brown


%% Initialize
clear all

% Constants
    % Topography Constants
    topomax = 6000; % high of valley in meters
    m = 1; % slope of valley
   
    % Mass Balance Constants
    gamma = 0.01; % b slope in yr-1
    ampELA=500; % half amplitude of sin function for varying climate, how much elevation does ELA change by +/-
    pELA = 250; % period of climate oscillations in years
    startELA = (topomax*(2/3)); % equilibrium line altitude (ELA) mid in meters
    ELA = startELA; % Initial ELA

    % Flux Constants
    Uo=60; % glacier sliding m/year
    Hstar = 60; % critial thickness in meters
    A = 10^-16; % Flow law parameter in Pa-3 yr-1
    rhoi = 917; % density of ice in kg/m3
    g = 9.81; % gravitational acceleration
    valleywidth = 200; % width of valley holding the glacier in meters
        
% Time and Space
    % Time Array
    dt = 0.01; % time step in years
    tmax = 1000; % tmax in years
    t = 0:dt:tmax; % time array
    imax = length(t); % loop length
    nplot = 100; % number of plots
    tplot = tmax/nplot; % times at which plots will be made

    % Space Array
    dx = 100; % horizontal step in meters
    xmin = 0; % min horizontal space in meters
    xmax = topomax/m; % max horizontal space in meters
    x = xmin:dx:xmax; % x array
    xsq = x.*x;
    xedge = xmin+(dx/2):dx:xmax-(dx/2); % xedge array

%Variable Arrays
    %Nodes
    N = ((xmax/dx))+1; % number of nodes, based on space array

    % Bedrock Topography
    %zrock = topomax - m.*x; % linear valley profile
    zrock = topomax*exp(-.5*m*10^-3*x);


    % Glacier Variables and Arrays
    icethickness = zeros(1,N); % glacier thickness, initial condition is no ice
    iceelevation = zrock+icethickness; % ice elevation array
    iceslope = diff(iceelevation)./dx; % array of slope based on ice elevations
    edgeicethickness = zeros(1,N-1); % array for ice thicknesses at the edges
    jmax = length(edgeicethickness); % length for loop to fill ice thickness at edges
    for j=1:1:jmax
        edgeicethickness(j) = (icethickness(j)+icethickness(j+1))/2; % calculate thickness of ice at edges by averaging boxes 
    end
        
    % Mass Balance
    b = gamma*(iceelevation-ELA); % mass balance, b, array
        
    % Ice Flux via Flow Law
    Qflow = zeros(1,N+1); % inital condition for ice flux by the flow law is no flux
        

% For Plotting Make Arrays of Specific Sizes

%For Plotting x Axis
xplot = zeros(1,N+1); %for use in elevation and thickness plots
xplot(1)=-1; %for use in elevation and thickness plots
xplot(2:N+1)=x; %for use in elevation and thickness plots
xplot2=xmin:dx:xmax+dx; %for use in Q plot

%For Plotting Ice Elevation
iceplot = zeros(1,N+1);
iceplot(1)=topomax;
iceplot(2:N+1)=iceelevation;

%For Plotting Bedrock Elevation
rockplot = zeros(1,N+1);
rockplot(1)=0;
rockplot(2:N+1)=zrock;

%For Plotting Ice Thicknesses
thicknessplot = zeros(1,N+1);
thicknessplot(1)=0;
thicknessplot(2:N+1)=icethickness;

%For Plotting Mass Balance Line
zplot =0:1000:10000; % array of large elevation values
zerosforb = zeros(size(zplot)); % make a line at x = 0 for y values in zplot

%For Filling Plots
bottomline1=zeros(N+1:2); %%Create a bottom line array for use in filling plots, needs to be same length as N
bottomline1(2:N+1)=-20000000; %arbitrary very negative number
bottomline2=zeros(N:2); %%Create a bottom line array for use in filling plots, needs to be same length as N
bottomline2(2:N)=-20000000; %arbitrary very negative number




        
%% Run
for i=1:1:imax
    
    % Determine ELA
    ELA = startELA + (ampELA*sin(2*pi*t(i)/pELA)); % varying ELA with time around a starting ELA value
    ELAx = ones(length(xplot))*ELA; %make an array the size of x to fill with the ELA this is for plotting a moving line of ELA on glacier profile
    
    % Update Mass Balance
    bzero = gamma.*(iceelevation - startELA); %the initial condition for the b line, to be plotted in mass balance plot
    b = gamma.*(iceelevation - ELA); % recalculates b based on the updated ELA array for model 
    
    % Calculate Ice Flux 
    iceslope = diff(iceelevation)./dx; % calculate slopes on ice surface      
    for j=1:1:jmax  %go through the x array
            edgeicethickness(j) = (icethickness(j)+icethickness(j+1))/2; % update the ice thickness at the edge of each dx element 
    end
    Uslide = Uo.*exp(-edgeicethickness./Hstar); %glacier sliding based on thickness of ice
    Qflow(2:N) = (Uslide.*edgeicethickness)-((1/valleywidth)*(A.*((rhoi*g*iceslope).^3).*((edgeicethickness.^5)/5))); % Ice Flux at each position according to flow law
    Qflow(1)=0; % no ice flux at the top, boundary condition
    Qflow(N+1)=0; % no ice flux at the bottom, boundary condition
    
    % Calculate Change in Ice Thickness, dhdt
    dQdx = diff(Qflow)./dx; % calculate net flux through each dx element
    dhdt = b - dQdx; % change in ice thickness for time step is mass from b, minus the net flux via the flow law
    
    % Update Total Ice Thickness
    icethickness = icethickness + (dhdt*dt); % update ice thickness with dhth
    Hneg = find(icethickness<0); % don't let the ice thickness be negative
    icethickness(Hneg)=0; % don't let the ice thickness be negative
    
    % Update Elevation of Ice
    iceelevation = zrock + icethickness; % update ice elevation
    
    % Make Plotting Easy
    iceplot(2:N+1)=iceelevation; % for plotting the ice elevation
    thicknessplot(2:N+1)=icethickness; % for plotting the ice thickness 

    
    % Plot results
    if rem(t(i),tplot)==0  
        figure(1)
        clf
        
        %Glacier Profile Subplot
            subplot('position',[.1 .45 .8 .5])
            plot(x,zrock,'k','linewidth',1)
            hold all
            
            plot(xplot,iceplot,'c')
            
            %Color the Glacier
            xx = [xplot xplot];
            yy = [iceplot bottomline1];
            fill(xx,yy,[.62 .82 .882]);
            
            %Color the Bedrock
            xx = [x x];
            yy = [zrock bottomline2];
            fill(xx,yy,[.91,.89,.87]);
            
            %Add the ELA Line
            plot(xplot,ELAx,'Color',[.812 .49 .392], 'linewidth', 3)
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',22)
            ylabel('Elevation (m)', 'fontname', 'arial', 'fontsize',22)
            title('Glacier Profile in Relation to ELA through Time')
            set(gca,'fontsize',18,'fontname','arial')
            ht=text(2500,6500,['  ',num2str(round(t(i))), ' yrs  '],'fontsize',20); %print time in animation
            axis([0 (1/2)*topomax 500 topomax+1000])
            hold all 
            
        %Mass Balance Subplot
            subplot('position',[.1 .1 .2 .25])
            plot(b,iceelevation,'Color',[.812 .49 .392], 'linewidth', 3)
            hold all
            
            %Plot Formatting
            plot(zerosforb,zplot, 'k--')
            plot(bzero,iceelevation,'k')
            xlabel('Accumulation/Ablation (m/yr)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Elevation (m)', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([-30 15 0 6000])
            hold all

        %Ice Thickness Subplot
            subplot('position',[.4 .1 .2 .25])
            fill(xplot,thicknessplot,[.62 .82 .882])
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Ice Thickness (m)', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([0 (1/2)*topomax 0 200])
            hold all 

        %Ice Flux Subplot
            subplot('position',[.7 .1 .2 .25])
            plot(xplot2,Qflow/10000,'Color',[.812 .392 .608], 'linewidth', 4)
            hold on
            
            %Plot Formatting
            xlabel('Distance (m)', 'fontname', 'arial', 'fontsize',16)
            ylabel('Ice Flux via Flow m^3/yr x 10^4', 'fontname', 'arial', 'fontsize',16)
            set(gca,'fontsize',18,'fontname','arial')
            axis([0 (1/2)*topomax 0 2])
    
        pause(0.1)
    end
end
       
        
        
        
        