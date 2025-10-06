--[[
	Artificial Neural Network Library 
	~By Kironte~
	
	Version: 1.04 (12/14/2019)
	UPDATE: 
	Added neural network visuals.
	
	Credit:
	If you are going to use this library, please disclose credit ingame or in the description. I would appreciate it!
	You are not to claim any of the following code as your own in its entirety. Modding and editing to make your own
	is completely fine as long as you note the original library.
	
	Documentation:
	All documentation is listed on the DevForum page on: 
	https://devforum.roblox.com/t/neural-network-library-artificial-neural-networks/400885
--]]



local module = {}

math.randomseed(tick())

function module.arrayCopy(original) --Function needed to copy arrays in Lua
    local copy = {}
    for i=1,#original do
        table.insert(copy,i,original[i])
    end
    return copy
end

function module.sigmoid(x,deriv) --For output functions we use a different activator
	if deriv then
		return x*(1-x)
	end
	return 1/(1+2.718281828459^(-x))
end

function module.activFunc(x,deriv,activator)
	if activator=="Identity" then  --I mean, if it works?
		if deriv then
			return 1
		end
		return x
	end
	if activator=="Binary" then  --l0l gl
		if deriv then
			return 0
		end
		if x >= 0 then
			return 1
		end
		return 0
	end
	if activator=="Sigmoid" then			--Good for tiny 1-3 layer networks, DO NOT USE for anything past that.
		if deriv then
			return x*(1-x)
		end
		return 1/(1+2.718281828459^(-x))
	end
	if activator=="Tanh" then				--Better than sigmoid but still will not work well past 6-7 layers.
		if deriv then
			return 1/math.cosh(x)^2
		end
		return math.tanh(x)
	end
	if activator=="ArcTan" then			
		if deriv then
			return 1/(x^2+1)
		end
		return math.atan(x)
	end
	if activator=="Sin" then 			--maybe...?
		if deriv then
			return math.cos(x)
		end
		return math.sin(x)
	end
	if activator=="Sinc" then 
		if deriv then
			if x==0 then
				return 0
			end
			return math.cos(x)/x-math.sin(x)/x^2
		end
		if x==0 then
			return 1
		end
		return math.sin(x)/x
	end
	if activator=="ArSinh" then 
		if deriv then
			return 1/(x^2+1)^0.5
		end
		return math.log(x+(x^2+1)^0.5)
	end
	if activator=="SoftPlus" then 
		if deriv then
			return 1/(1+2.718281828459^(-x))
		end
		return math.log(1+2.718281828459^x)
	end
	if activator=="BentIdentity" then 
		if deriv then
			return x*(2*(x^2+1)^0.5)+1
		end
		return ((x^2+1)^0.5-1)/2+x
	end
	if activator=="ReLU" then				--Works well for all layer counts but tends to kill neurons, resulting in bad networks.
		if deriv then
			if x>0 then
				return 1
			elseif x==0 then
				return 0.5
			end
			return 0
		end
		return math.max(0,x)
	end
	if activator=="SoftReLU" then			--Works well for all layer counts but still tends to give unstable scores.
		if deriv then
			return 1/(1+2.718281828459^(-x))
		end
		return math.log(1+2.718281828459^(x))
	end
	if activator=="LeakyReLU" then			--Works well for all layer counts and doesn't kill too many neurons. Best function overall.
		if deriv then
			if x >= 0 then
				return 1
			end
			return 0.1
		end
		return math.max(0.1*x,x)
	end
	if activator=="Swish" then
		if deriv then
			return (2.718281828459^(-x)*(x+1)+1)/(1/(1+2.718281828459^(-x)))^2
		end
		return x*(1/(1+2.718281828459^(-x)))
	end
	if activator=="ElliotSign" then
		if deriv then
			return 1/(1+math.abs(x))^2
		end
		return x/(1+math.abs(x))
	end
	if activator=="Gaussian" then 
		if deriv then
			return -2*x*2.718281828459^(-x^2)
		end
		return 2.718281828459^(-x^2)
	end
	if activator=="SQ-RBF" then   --god why have you forsaken us
		if deriv then
			if math.abs(x)<=1 then 
				return -x
			end
			if 1<=math.abs(x) and math.abs(x)<2 then
				return 2-x
			end
			return 0
		end
		if math.abs(x)<=1 then --No point in wasting power for else statements when working with lots of returns
			return 1-x^2/2
		end
		if 1<=math.abs(x) and math.abs(x)<2 then
			return 2-(2-x^2)/2
		end
		return 0 
	end
	error("Activator unsupported. Please refer to documentation for supported activation functions.")
end

function module.layerMath(input,layerArray,lastLayer, active) 
	local weights,biases,isRecurrent = layerArray[1],layerArray[2],layerArray[3]
	local output = {}
	local func = module.activFunc
	if lastLayer then
		func=module.sigmoid   --SoftMax here I come!
	end
	for i=1, #biases do
		local sum = 0
		for d=1, #input do
			sum = sum + input[d]*weights[i][d]
		end
		if isRecurrent then
			sum = sum + isRecurrent[i]*weights[i][#input+1] --Add in the last timestep's output for recurrency
		end
		sum = sum + biases[i]
		table.insert(output,func(sum,false,active))
	end
	return output
end

function module.createNet(inputs, hiddenL, hiddenN, outputs, active, recurrent, defBias, warning)
	--{inputs,{layerArray1,layerArray2,outputArray},best,activator}
	--layerArrays = {weightArray, biasArray} 			If Recurrent = {weightArray, biasArray, recurrentArray} 
	--weightArray = {nodeWeightArray,nodeWeightArray}  	If Recurrent = {nodeWeightArray+recurrentWeight,nodeWeightArray+recurrentWeight}
	--biasArray = {0,0}
	--recurrentArray = {0,0}
	--best = false
	--activator = "LeakyReLU"
	
	--Data Validation-------------------
	if type(inputs)~="number" then error("Wrong Arg#1: Input Count must be an integer.") end
	if type(hiddenL)~="number" then error("Wrong Arg#2: Hidden Layer Count must be an integer.") end
	if type(hiddenN)~="number" then error("Wrong Arg#3: Hidden Node Count must be an integer.") end
	if type(outputs)~="number" then error("Wrong Arg#4: Output Count must be an integer.") end
	if type(active)~="string" and active~=nil then error("Wrong Arg#5: Activator Function must be string.") end
	if type(recurrent)~="boolean" and recurrent~=nil then error("Wrong Arg#6: Recurrent must be a boolean.") end
	if type(defBias)~="number" and defBias~=nil then error("Wrong Arg#7: Default Bias must be a decimal.") end
	if not defBias then
		defBias = 0.5 --Default bias
	end
	if not active then
		active = "LeakyReLU" --Default activator
	end
	if recurrent and not warning then
		if active~="Sigmoid" and active~="Tanh" then
			warn("!WARNING!: '"..active.."' is NOT supported for Recurrent Network backpropagation. Expect problems!")
		end
	end
	-------------------------------------
	local layersArray = {}
	for i=1, hiddenL+1 do
		local layerArray,nodeCount,weightCount = {{},{}},hiddenN,hiddenN
		
		if i>hiddenL then
			nodeCount = outputs
		end
		if i==1 then
			weightCount = inputs
		end
		if recurrent then
			weightCount = weightCount + 1  --recurrentWeight
			table.insert(layerArray,{})
		end
		
		for z=1, nodeCount do		
			local weight = {}
			for z=1, weightCount do	
				table.insert(weight,math.random(-200000000,200000000)/100000000*math.sqrt(2/nodeCount)) --'He' Weight initialization method used, precision of 10^-8
			end
			table.insert(layerArray[1],weight)
			table.insert(layerArray[2],defBias)
			if recurrent then
				table.insert(layerArray[3],0)
			end
		end
		
		table.insert(layersArray,layerArray)
	end
	return {inputs,layersArray,false,active}	
end

function module.forwardNet(net, a, giveCache) --forward propogation
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	if type(a)~="table" then error("Wrong Arg#2: Inputs must be in an array.") end
	if type(giveCache)~="boolean" and giveCache~=nil then error("Wrong Arg#3: Must be a Boolean.") end
	-------------------------------------
	
	
    local inputCount,layersArray = net[1],net[2]
	local isRecurrent = layersArray[1][3]
	
    if inputCount ~= #a then print("Incorrect input count.") return end
    local cache = {}
    for i=1, #layersArray-1 do
        a = module.layerMath(a,layersArray[i],false,net[4])
		if isRecurrent then
			layersArray[i][3] = a --If the network is recurrent, we save the outputs of the current layer for the next timestep
		end
        table.insert(cache,a)     
    end
    a = module.layerMath(a, layersArray[#layersArray], true)
    table.insert(cache, a)
	if giveCache then
    	return {a,cache}
	end
	return a
end

function module.hardCode (array)
	--Data Validation-------------------
	if type(array)=="userdata" then
		if array.ClassName=="StringValue" then
			array = module.loadNet(array.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(array)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	-------------------------------------
	local output = "{"	--Same as JSONEncode but uses curly brackets, allowing you to copy paste this into a script
	for i=1, #array do
		if type(array[i])=="table" then
			output = output..module.saveNet(array[i])
		else
			output = output..tostring(array[i])
		end
		if i ~= #array then
			output = output..","
		end
	end
	return output.."}"
end

function module.saveNet (array) 
	--Data Validation-------------------
	if type(array)=="userdata" then
		if array.ClassName=="StringValue" then
			array = module.loadNet(array.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(array)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	-------------------------------------
	return game:GetService("HttpService"):JSONEncode(array) --For strictly Roblox, this is the preferable method
end

function module.loadNet (str)
	--Data Validation-------------------
	if type(str)=="userdata" then
		if str.ClassName=="StringValue" then
			str = str.Value
		else
			print("Wrong Arg#1: Network must be a string or StringValue.")
		end
	else
		if type(str)~="string" then
			error("Wrong Arg#1: Network must be a string or StringValue.")
		end
	end
	-------------------------------------
	return game:GetService("HttpService"):JSONDecode(str)
end

function module.backwardNet(net,rate,input,target)
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	if type(rate)~="number" then error("Wrong Arg#2: Rate must be a decimal.") end
	if type(input)~="table" then error("Wrong Arg#3: Inputs must be in an array.") end
	if type(target)~="table" then error("Wrong Arg#4: Targets must be in an array.") end
	-------------------------------------
	
	local activator = net[4]
	local isRecurrent = net[2][1][3]
	local test = module.forwardNet(net,input,true)
	local layersArray = net[2]
	local output,activations = test[1],test[2]  --Get the current output and activations cache for the given input
	local delta = {}
	local layersArrayC = #layersArray --To avoid unneeded recalculation, we make Lua calculate this number beforehand. We don't use the variable
	
	delta[#layersArray] = {}
	local layer = layersArray[#layersArray]
	for d=1, #layer[2] do                    --Output nodes use a different formula so, to avoid extra ifs, we just hardcode this one
		local nodeOutput = activations[#layersArray][d]
		delta[#layersArray][d] = module.sigmoid(nodeOutput,true)*(nodeOutput-target[d]) --Output only uses sigmoid so we stick with that
	end
	
	for i=#layersArray-1, 1, -1 do	--Loop through every layer backwards
		--delta = {layerArray,layerArray}
		--layerArray = {nodeGradient,nodeGradient}
		
		delta[i] = {}
		local layer = layersArray[i]
		for d=1, #layer[2] do 
			local nodeOutput = activations[i][d]
			local sum = 0
			local afterLayer = layersArray[i+1] 
			local afterLayerDelta = delta[i+1]
			
			for z=1, #afterLayerDelta do
				sum = sum + afterLayerDelta[z]*afterLayer[1][z][d]
			end
			delta[i][d] = module.activFunc(nodeOutput,true, activator)*sum  --Taking and saving the gradient
		end
	end
	local layer = layersArray[1]  --In order to avoid an unneeded if statement, we hardcode an iteration of the loop below VV for the first layer
	for d=1, #layer[2] do --Up to number of biases (nodes)
		local gradient = delta[1][d] --Take node's gradient
		local limit = #layer[1][d]
		if isRecurrent then
			
			layer[1][d][limit] = layer[1][d][limit] + -rate * delta[1][d] * layer[3][d]
			limit = limit - 1
		end
		for z=1, limit do 	--For every weight in d'th node
			layer[1][d][z] = layer[1][d][z] + -rate * delta[1][d] * input[z]
		end
		layer[2][d] = layer[2][d] -rate * gradient
	end
	for i=2, #layersArray do	 																						--<<
		local layer = layersArray[i]
		for d=1, #layer[2] do
			local gradient = delta[i][d]
			local limit = #layer[1][d]
			if isRecurrent then
				layer[1][d][limit] = layer[1][d][limit] + -rate * delta[i][d] * layer[3][d]
				limit = limit - 1
			end
			for z=1, limit do
				layer[1][d][z] = layer[1][d][z] + -rate * delta[i][d] * activations[i-1][z]  
			end
			layer[2][d] = layer[2][d] -rate * gradient
		end
	end
end

--Function for creating a genetic network in the chosen folder with the chosen parameters. Used as a basis for the genetic algorithm functions
--Returns an array with the entries that contain the saved networks so we do not have to :GetChildren()
function module.createGenNet (folder, networkCount, inputs, hiddenL, hiddenN, outputs, active, recurrent, defBias)
	--Data Validation-------------------
	if type(folder)~="userdata" then error("Wrong Arg#1: Container must be an object.") end 
	if type(networkCount)~="number" then error("Wrong Arg#2: Network Count must be an integer.") end
	if type(inputs)~="number" then error("Wrong Arg#3: Input Count must be an integer.") end
	if type(hiddenL)~="number" then error("Wrong Arg#4: Hidden Layer Count must be an integer.") end
	if type(hiddenN)~="number" then error("Wrong Arg#5: Hidden Node Count must be an integer.") end
	if type(outputs)~="number" then error("Wrong Arg#6: Output Count must be an integer.") end
	if type(active)~="string" and active~=nil then error("Wrong Arg#7: Activator Function must be string.") end
	if type(recurrent)~="boolean" and recurrent~=nil then error("Wrong Arg#8: Recurrent must be a boolean.") end
	if type(defBias)~="number" and defBias~=nil then error("Wrong Arg#9: Default bias must be a number.") end
	-------------------------------------
	for i=1, networkCount do
		local network = module.saveNet(module.createNet(inputs,hiddenL,hiddenN,outputs, active, recurrent, defBias, true))
		local entry = Instance.new("StringValue")
		entry.Value = network
		entry.Name = "GenNet_"..i
		entry.Parent = folder
	end
	return folder:GetChildren()
end

function module.runGenNet (nets, scores, giveBest)
	--Data Validation-------------------
	if type(nets)=="userdata" then
		nets = nets:GetChildren()
		if #nets<3 then
			print("Wrong Arg#1: Networks must be an array or container (folder?).")
		end
	else
		if type(nets)~="table" then
			error("Wrong Arg#1: Networks must be an array or container (folder?).")
		end
	end
	if type(nets)~="table" then error("Wrong Arg#1: Network references must be in an array (GetChildren).") end
	if type(scores)~="table" then error("Wrong Arg#2: Scores must be in an array.") end
	-------------------------------------
	local isRecurrent = module.loadNet(nets[1])[2][1][3]~=nil --Are the networks recurrent?
	local active = module.loadNet(nets[1])[4] --Getting the activator used
	local scoresCache = module.arrayCopy(scores) --Variable that contains the unchanging scores, used for the breeding chance loop
	local bestScore
	local dead = {}
	local best = {}
	local breeders = math.max(math.ceil(#nets*0.4),2)   --We breed only the top 40%, since we need at least 2 to breed and another to fill in with a child, the minimum network count is 3
	for i=1,breeders do        		--Finding the best 2 networks according to the given scores
		local top = {0,0}
		for d=1,#scoresCache do
			if scoresCache[d]>top[1] then
				top[1],top[2]=scoresCache[d],d
			end
		end
		scoresCache[top[2]] = -math.huge
		table.insert(best,top[2])
	end
	local noMutation = module.loadNet(nets[best[1]]) --Take the best network and make it exempt from any mutation for this cycle. This is to combat degeneration.
	noMutation[3] = true
	nets[best[1]].Value = module.saveNet(noMutation)
	local kills = math.ceil((#nets-breeders)*0.5)   --We kill 75% of the losers while making sure to kill at least 1
	for i=1, kills do        		--Finding the best 2 networks according to the given scores
		local bottom = {math.huge,0}
		for d=1,#scoresCache do
			if scoresCache[d]<bottom[1] and scoresCache[d]~=-math.huge then
				bottom[1],bottom[2] = scoresCache[d],d
			end
		end
		scoresCache[bottom[2]] = math.huge
		nets[bottom[2]].Value = ""
	end
	for i=1,#nets do
		if nets[i].Value == "" then
			table.insert(dead,i)
		end
	end
	local chances = {}
	local min,max = scores[best[#best]],scores[best[1]]
	local range = max-min
	for i=1, #best do
		local chance = (scores[best[i]]-min)/range*0.8+0.1  --Assigns the top networks a % chance of breeding from 75% to 25%
		table.insert(chances,chance*100)
	end
	local noiseRange = 0.01 --No child is perfect. All crossed out parameters have a tiny noise or mutation applied to make them less identical
	for i=1, kills do
		local parent1,parent2 = 1,2
		if #best > 2 then
			parent1,parent2 = math.random(1,#best),math.random(1,#best)
			while math.random(0,100) <= chances[parent1] do
				parent1 = math.random(1,#best)
			end
			while parent1 == parent2 or math.random(0,100) <= chances[parent2] do
				parent2 = math.random(1,#best)
			end
		end
		local parent1,parent2 = module.loadNet(nets[best[parent1]]),module.loadNet(nets[best[parent2]])
		local val1,val2,val3,val4,val5,val6 = module.getNetworkData(parent1)
		local child = module.createNet(val1,val2,val3,val4,val5,val6, nil, true)
		for d=1, #child[2] do
			for z=1, #child[2][d][2] do
				if math.random(0,1)==1 then
					if isRecurrent then
						child[2][d][3][z] = parent1[2][d][3][z]
					end
					child[2][d][2][z] = parent1[2][d][2][z] + math.random(-noiseRange*100000,noiseRange*100000)/100000
					child[2][d][1][z] = parent1[2][d][1][z] --+ math.random(-noiseRange*100000,noiseRange*100000)/100000
					
				else
					if isRecurrent then
						child[2][d][3][z] = parent2[2][d][3][z]
					end
					child[2][d][2][z] = parent2[2][d][2][z] + math.random(-noiseRange*100000,noiseRange*100000)/100000
					child[2][d][1][z] = parent2[2][d][1][z] --+ math.random(-noiseRange*100000,noiseRange*100000)/100000
				end
				
			end
		end
		nets[dead[1]].Value = module.saveNet(child)
		table.remove(dead,1)
	end
	local mutate = 2 --Number of nodes to mutate in each network
	local mutateRange = 4 --Range for the mutation value above and below 0
	for i=1, #nets do
		local net = module.loadNet(nets[i])
		if net[3] == false then
			local nodes = {}
			for d=1, mutate do
				while true do   --Needed infinite while loop to make sure that different nodes are chosen
					local layer = math.random(1,#net[2])
					local node = math.random(1,#net[2][layer][2])
					local same = false
					for z=1, #nodes do
						if nodes[z][1] == layer and nodes[z][2] == node then
							same = true
						end
					end
					if not same then
						table.insert(nodes,{layer,node})
						break
					end
				end
			end
			for d=1, #nodes do
				local node = nodes[d]
				local weights = net[2][node[1]][1][node[2]]
				for z=1, #weights do
					net[2][node[1]][1][node[2]][z] = net[2][node[1]][1][node[2]][z] + math.random(-mutateRange*100000,mutateRange*100000)/100000
				end
				net[2][node[1]][2][node[2]] = net[2][node[1]][2][node[2]] + math.random(-mutateRange*100000,mutateRange*100000)/100000
			end
			nets[i].Value = module.saveNet(net)
		else
			net[3] = false --Making sure that the best network is safe from mutations only for this cycle
			nets[i].Value = module.saveNet(net)
		end
	end
	return best[2]
end

function module.getNetworkData(net)
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	------------------------------------
	return net[1],#net[2]-1,#net[2][1][2],#net[2][#net[2]][2],net[4],net[2][1][3]~=nil
end

function module.getAproxLength(inputs, hiddenL, hiddenN, outputs, active, recurrent)
	--Data Validation-------------------
	if type(inputs)~="number" then error("Wrong Arg#1: Input Count must be an integer.") end
	if type(hiddenL)~="number" then error("Wrong Arg#2: Hidden Layer Count must be an integer.") end
	if type(hiddenN)~="number" then error("Wrong Arg#3: Hidden Node Count must be an integer.") end
	if type(outputs)~="number" then error("Wrong Arg#4: Output Count must be an integer.") end
	if type(active)~="string" and active~=nil then error("Wrong Arg#5: Activator Function must be string.") end
	if type(recurrent)~="boolean" and recurrent~=nil then error("Wrong Arg#6: Recurrent must be a boolean.") end
	-------------------------------------
	local numberCost = 31 --Average number of characters in each parameter
	local size = 0
	local cache = {inputs}
	for i=1,hiddenL do
		table.insert(cache,hiddenN)
	end
	table.insert(cache,outputs)
	for i=2, #cache do
		size = size + (cache[i-1]*cache[i]+cache[i])*numberCost --Parameter sizes
		if recurrent then
			size = size + (cache[i-1]+cache[i])*numberCost --Recurrent weights and activations
		end
		size = size + cache[i]+1 + (cache[i-1]+1)*cache[i] --Commas and brackets
	end 
	return size + #tostring(inputs) + 8 + #active
end

function module.getVisual(net)
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	------------------------------------
	local inputs,hiddenL,hiddenN,outputs,isRecurrent = module.getNetworkData(net)
	local layerCount = hiddenL+2
	local screen = Instance.new("ScreenGui")
	screen.Name = "NNVisualizer"
	
	local container = Instance.new("Frame")
	container.Name = "Container"
	container.Position = UDim2.new(0.015,0,0.039,0)
	container.Size = UDim2.new(0,549,0,290)
	container.BackgroundTransparency = 0.75
	container.Parent = screen
	
	local aspectRatio = Instance.new("UIAspectRatioConstraint")
	aspectRatio.AspectRatio = 16/9
	aspectRatio.Parent = container
	
	local nodeCount = {inputs}
	for i=1, hiddenL do
		table.insert(nodeCount,hiddenN)
	end
	table.insert(nodeCount,outputs)
	
	local layerWidth = 0.1*4/(layerCount)
	local nodePadding = 0.02
	local mostNodes = math.max(inputs,hiddenN,outputs)
	
	--If the nodes are overflowing the container VV
	if mostNodes*layerWidth*(container.AbsoluteSize.X/container.AbsoluteSize.Y)+nodePadding*(mostNodes-1)>1 then
		layerWidth = (1-nodePadding*(mostNodes-1))/(mostNodes*(container.AbsoluteSize.X/container.AbsoluteSize.Y))
	end
	
	for i=1, #nodeCount do
		local layer = Instance.new("Frame")
		layer.Name = "Layer"..i
		layer.AnchorPoint = Vector2.new(0.5,0.5)
		layer.BackgroundTransparency = 1
		layer.Size = UDim2.new(layerWidth,0,1,0)
		layer.Position = UDim2.new((layer.Size.X.Scale/2)+(1-layer.Size.X.Scale)*((i-1)/(#nodeCount-1)),0,0.5,0)
		layer.Parent = container
		
		local nodeLayout = Instance.new("UIListLayout")
		nodeLayout.Name = "NodeLayout"
		nodeLayout.Padding = UDim.new(nodePadding,0)
		nodeLayout.FillDirection = "Vertical"
		nodeLayout.SortOrder = "LayoutOrder"
		nodeLayout.VerticalAlignment = "Center"
		nodeLayout.Parent = layer
			
		for d=1, nodeCount[i] do
			local node = Instance.new("ImageLabel")
			node.Name = "Node"..d
			node.LayoutOrder = d
			node.AnchorPoint = Vector2.new(0.5,0.5)
			node.BackgroundTransparency = 1
			node.Size = UDim2.new(1,0,0.186,0)
			node.Image = "http://www.roblox.com/asset/?id=130424513"
			node.ZIndex = 2
			node.Parent = layer
			
			local nodeAspect = Instance.new("UIAspectRatioConstraint")
			nodeAspect.Name = "NodeAspect"
			nodeAspect.AspectRatio = 1
			nodeAspect.AspectType = "ScaleWithParentSize"
			nodeAspect.DominantAxis = "Width"
			nodeAspect.Parent = node
		end
		
		if i~=1 then
			local synapses = Instance.new("Frame")
			synapses.Name = "Synapse"..i
			synapses.Size = layer.Size
			synapses.BackgroundTransparency = 1
			synapses.AnchorPoint = Vector2.new(0.5,0.5)
			synapses.Position = UDim2.new((container:FindFirstChild("Layer"..i-1).Position.X.Scale+layer.Position.X.Scale)/2,0,0.5,0)
			synapses.Parent = container
			for d=1, nodeCount[i] do
				for f=1, nodeCount[i-1] do
					local synapse = Instance.new("Frame")
					synapse.Name = "Syn"..d.."_"..f
					synapse.AnchorPoint = Vector2.new(0.5,0.5)
					local node1,node2 = layer["Node"..d],container["Layer"..i-1]["Node"..f]
					local diffX,diffY = node1.AbsolutePosition.X-node2.AbsolutePosition.X,node1.AbsolutePosition.Y-node2.AbsolutePosition.Y
					local mag = ((diffX^2+diffY^2)^0.5) / layer.AbsoluteSize.X
					local yPoint = (node1.AbsolutePosition.Y+node2.AbsolutePosition.Y)/2
					yPoint = yPoint/container.AbsoluteSize.Y
					synapse.Position = UDim2.new(0.5,0,yPoint-0.0035*mostNodes,0)
					synapse.Size = UDim2.new(mag,0,synapses.AbsoluteSize.X/synapses.AbsoluteSize.Y/10,0)
					synapse.Rotation = math.deg(math.atan(diffY/diffX))
					synapse.BorderSizePixel = 0
					synapse.Parent = synapses
				end
			end
		end
	end
	module.updateVisualState(net,screen)
	return screen
end

function module.updateVisualState(net,vis)
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	if type(vis)~="userdata" then error("Wrong Arg#2: Visual must be an object.") end
	------------------------------------
	local weightRange = {}
	local biasRange = {}
	for l=2, #net[2]+1 do
		local layer = net[2][l-1]
		for n=1, #layer[2] do
			local bias,weights=layer[2][n],layer[1][n]
			table.insert(biasRange,math.abs(bias))
			for w=1, #weights do
				local weight = weights[w]
				table.insert(weightRange,math.abs(weight))
			end
		end
	end
	table.sort(biasRange)	table.sort(weightRange)
	weightRange,biasRange = math.max(weightRange[#weightRange],3),math.max(biasRange[#biasRange],3)
	local container = vis.Container
	local uiLayer = container.Layer1
	for i=1, #uiLayer:GetChildren()-1 do
		local uiNode = uiLayer["Node"..i]
		uiNode.ImageColor3 = Color3.fromHSV(0,0,1)
	end
	for l=2, #net[2]+1 do
		local layer = net[2][l-1]
		local uiLayer = container["Layer"..l]
		local uiSyns = container["Synapse"..l]
		for n=1, #layer[2] do
			local bias,weights=layer[2][n],layer[1][n]
			local uiNode = uiLayer["Node"..n]
			uiNode.ImageColor3 = Color3.fromHSV(math.min(1,math.max(0,(bias+biasRange)/(biasRange*2)))*0.3,1,1)
			for w=1, #weights do
				local weight = weights[w]
				local uiSyn = uiSyns["Syn"..n.."_"..w]
				uiSyn.BackgroundColor3 = Color3.fromHSV(math.min(1,math.max(0,(weight+weightRange)/(weightRange*2)))*0.3,1,1)
			end
		end
	end
end

function module.updateVisualActive(net,vis,inputs,range)
	--Data Validation-------------------
	if type(net)=="userdata" then
		if net.ClassName=="StringValue" then
			net = module.loadNet(net.Value)
		else
			print("Wrong Arg#1: Network must be an array or StringValue.")
		end
	else
		if type(net)~="table" then
			error("Wrong Arg#1: Network must be an array or StringValue.")
		end
	end
	if type(vis)~="userdata" then error("Wrong Arg#2: Visual must be an object.") end
	if type(inputs)~="table" then error("Wrong Arg#3: Inputs must be in an array.") end
	if type(range)~="number" and range~=nil then error("Wrong Arg#4: Range must be an integer.") end
	------------------------------------
	if not range then
		range = 1
	end
	local nodeActives = module.forwardNet(net,inputs,true)[2]
	local synActives = {}
	for l=1, #net[2] do
		local layer = net[2][l]
		local synLayerActive = {}
		for n=1, #layer[2] do
			local weights=layer[1][n]
			local synWeightActive = {}
			for w=1, #weights do
				local weight = weights[w]
				if l==1 then
					table.insert(synWeightActive,weight*inputs[w])
				else
					table.insert(synWeightActive,weight*nodeActives[l-1][w])
				end
			end
			table.insert(synLayerActive,synWeightActive)
		end
		table.insert(synActives,synLayerActive)
	end
	local container = vis.Container
	local uiLayer = container.Layer1
	for i=1, #inputs do
		local uiNode = uiLayer["Node"..i]
		uiNode.ImageColor3 = Color3.fromHSV(0,0,math.min(1,math.max(0.5,(inputs[i]+range)/(range*2))))
	end
	for l=1, #synActives do
		local layer = net[2][l]
		local uiLayer = container["Layer"..l+1]
		local uiSyns = container["Synapse"..l+1]
		for n=1, #synActives[l] do
			local uiNode = uiLayer["Node"..n]
			uiNode.ImageColor3 = Color3.fromHSV(0,0,math.min(1,math.max(0.5,(nodeActives[l][n]+range)/(range*2))))
			for w=1, #synActives[l][n] do
				local uiSyn = uiSyns["Syn"..n.."_"..w]
				uiSyn.BackgroundColor3 = Color3.fromHSV(0,0,math.min(1,math.max(0.5,(synActives[l][n][w]+range)/(range*2))))
			end
		end
	end
end
return module
