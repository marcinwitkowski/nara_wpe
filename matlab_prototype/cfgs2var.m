% ============================================================================= 
% Load Parameters
% ============================================================================= 
if ischar(cfgs)
    run(cfgs);
else
    varnames = fieldnames(cfgs);
    for ii = 1 : length(varnames)
        eval([varnames{ii}, '= getfield(cfgs, varnames{ii});']);
    end 
end