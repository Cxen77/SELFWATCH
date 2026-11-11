"""Optional CUDA graph replay for a fixed, warmed, single-owner OSNet encoder."""
import torch


class GraphedReID(torch.nn.Module):
    """Reuse kernel launch sequences, never embeddings or person observations.

    The extractor owns one serialized CUDA stream and copies results to host
    before the next call. All graph inputs/outputs stay alive with this module.
    Unsupported batch shapes use the unchanged model. CPU uses it unchanged.
    """
    def __init__(self,model,device,half,stream,batch_sizes=(1,2,4,8,16),input_hw=(256,128)):
        super().__init__();self.model=model;self.graphs={}
        if model.training:raise ValueError('CUDA graph ReID requires an eval-mode model')
        if device.type!='cuda' or stream is None:return
        dtype=torch.float16 if half else torch.float32
        with torch.inference_mode(),torch.cuda.stream(stream),torch.autocast(device_type='cuda',enabled=half):
            for count in batch_sizes:
                inputs=torch.zeros((count,3,*input_hw),dtype=dtype,device=device)
                for _ in range(3):model(inputs)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):outputs=model(inputs)
                self.graphs[count]=(inputs,outputs,graph)

    def forward(self,batch):
        state=self.graphs.get(batch.shape[0])
        if state is None:return self.model(batch)
        inputs,outputs,graph=state
        inputs.copy_(batch)
        graph.replay()
        return outputs
