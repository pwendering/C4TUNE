function [simA] = simulate_AQ(Variable,KE_type)
% This function simulate AQ curves under changing PAR with constant Air CO2 (Ca)
%
% INPUTS:
%    Variable:              Kinetic parameters
%
%
% OUTPUT:
%    simA:                  Simulated photosynthetic rate


%% Leaf gas exchange measurements 
% Stabilization of A and gs at the beginning of measurements with 400 CO2
% and PAR1800, followed by stepwise decrease until PAR=50


Q_t = configC4Sim('Q_t');

% Set empty vector for Ca changes
Ca_t=[];

% Set experimental condition and load global variables for C4 model
air_RH=0.65;
air_temp=25;

% Load metabolite names
mets_name=load_metabolite_names();
[~,ind_gs]=ismember("Gs[Leaf]",mets_name);

% index of A in Gs_VEL
ind_A = 2;

% Load initial concentrations
envFactor=var_env(Q_t(1),Ca_t,air_temp,air_RH,[]);
Ini=RAC4leafMetaIni(envFactor);% Initial values
nm=length(Ini);

%% Start ACi record
% global Gs_VEL initialized in optim_initialization...
nchange=length(Q_t);

% Initialize simulation vectors
simA=zeros(nchange,1);

continue_flag=true;
i=1;
while continue_flag && i<(nchange+1)

    ode_sol=[];
    initialization_flux_output();
    global Gs_VEL
    if i==1
        xt0=Ini;
    else
        xt0=current_state;
    end

    if i==1
        startTime = datetime('now');
        stop_fun = @(t,y)reaching_steadyA(t,y,startTime);
        tspan=[0 60*60]; % 60 minutes for stabilization
        options=odeset('NonNegative',1:nm, 'RelTol', 1e-04,'Events',stop_fun); % stop event is steady state of A
    else
        startTime = datetime('now');
        stop_fun = @(t,y)timeLimit(t,y,startTime);
        tspan=[0 2*60];  % 120 second time interval between record
        options=odeset('NonNegative',1:nm, 'RelTol', 1e-04,'Events',stop_fun); 
    end

    envFactor=var_env(Q_t(i),Ca_t,air_temp,air_RH,[]);

    ode_sol=ode15s(@(t,x)RAC4leafMetaMB(t,x,Variable,KE_type,envFactor),tspan,xt0,options); 
    
    if i==1 
        if isempty(ode_sol.xe) || max(ode_sol.xe)==tspan(2) || Gs_VEL(end,ind_A)<1
            continue_flag=false;
        end
    else
        if isempty(ode_sol) || max(ode_sol.x)~=tspan(2)
            continue_flag=false;
        end
    end
    if continue_flag==true
        simA(i)=Gs_VEL(end,ind_A);
        current_state=ode_sol.y(:,end);
    else
        simA(i)=1e10;
    end
    i=i+1;
end
% simA=real(simA);


