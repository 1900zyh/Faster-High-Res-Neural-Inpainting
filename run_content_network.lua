require 'torch'
require 'nn'
require 'optim'
require 'image'
require 'nngraph'
require 'cudnn'
util = paths.dofile('util.lua')
torch.setdefaulttensortype('torch.FloatTensor')

opt = {
  gpu=1,
  fineSize=128,
  overlapPred=4,
  model_file='models/imagenet_inpaintCenter.t7',
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local function loadImage(path,size)
    local input = image.load(path, 3, 'float')
    input = image.scale(input,size,size)
    input:mul(2):add(-1)
    return input
end

modelG=util.load(opt.model_file,opt.gpu)
modelG:evaluate()

all_fnames = paths.dir('examples')
fnames = {}
for i=1,#all_fnames do
  if string.match(all_fnames[i], "fake") or string.match(all_fnames[i], "demo") then ;
  else 
    if all_fnames[i] == '.' or all_fnames[i] == '..' then ;
    else
      table.insert(fnames, all_fnames[i])
    end
  end  
end
print(string.format('testing %d images ...', #fnames))

for i=1,#fnames do
  print(i, fnames[i])
  local real=torch.Tensor(1,3,512,512)
  local real_ctx = torch.Tensor(1,3,128,128)
  local fake2 = torch.Tensor(3,256,256)
  local output=torch.Tensor(3,512,512)
  real[1]=loadImage(string.format('examples/%s',fnames[i]),512)
  real_ctx[1]:copy(image.scale(real[1],128,128))
  real_ctx[{1,{1},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*117/255.0 - 1.0
  real_ctx[{1,{2},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*104/255.0 - 1.0
  real_ctx[{1,{3},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred},{1 + opt.fineSize/4 + opt.overlapPred, opt.fineSize/2 + opt.fineSize/4 - opt.overlapPred}}] = 2*123/255.0 - 1.0
  fake = modelG:forward(real_ctx)

  fake2:copy(image.scale(fake[1],256,256))
  output:copy(real[1])
  output[{{},{145,368},{145,368}}]:copy(fake2[{{},{17, 240},{17, 240}}])
  output[output:gt(1)]=1
  output[output:lt(-1)]=-1
  output=(output+1)/2
  image.save(string.format('examples/fake_%s',fnames[i]),output)
end


