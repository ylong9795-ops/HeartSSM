"""GPU discovery and measured resource admission, before parent CUDA context.

This module intentionally imports only the standard library. A short-lived
probe must measure the actual forward/backward/native-slice workload. Merely
finding an idle utilization counter never admits a device.
"""
import csv
import io
import os
import subprocess
import time


class WaitingForGPUResources(RuntimeError):
    pass


def _query(fields, kind='gpu'):
    result = subprocess.run(['nvidia-smi', f'--query-{kind}='+','.join(fields),
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode:
        raise WaitingForGPUResources('WAITING_FOR_GPU_RESOURCES: '+result.stderr.strip())
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def discover_gpus():
    rows = _query(('index','uuid','name','memory.total','memory.used','memory.free','utilization.gpu'))
    processes = _query(('gpu_uuid','pid','process_name','used_gpu_memory'), 'compute-apps')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    tokens = visible.split(',') if visible is not None else None
    devices = []
    for row in rows:
        index, uuid, name, total, used, free, utilization = [v.strip() for v in row]
        if tokens is not None and not any(token.strip() == index or uuid.startswith(token.strip())
                                          for token in tokens if token.strip()):
            continue
        local = (next(i for i,token in enumerate(tokens)
                      if token.strip() == index or uuid.startswith(token.strip())) if tokens is not None else int(index))
        devices.append(dict(physical_index=int(index), uuid=uuid, name=name, visible_cuda_index=local,
            total_mib=float(total), used_mib=float(used), free_mib=float(free), utilization=float(utilization),
            processes=[dict(pid=int(p[1]), name=p[2], used_mib=float(p[3]) if p[3].strip().isdigit() else None)
                       for p in processes if p[0].strip() == uuid]))
    return devices


def select_gpu(requested, probe, *, margin_fraction=.35, retries=0, retry_seconds=30., discover=discover_gpus):
    """probe(device) returns measured peak_reserved_mib + context_mib, or raises.

    No workload is started until the fresh free-memory observation satisfies
    the measured peak plus safety margin. Probe OOM cannot authorize CPU fallback.
    """
    attempts = []
    if requested == 'cpu':
        raise ValueError('This protocol does not silently run optimization on CPU')
    if margin_fraction <= 0 or retries < 0 or not 0 < retry_seconds <= 60:
        raise ValueError('Declare positive safety margin and finite resource retry settings')
    for retry in range(retries+1):
        devices = discover()
        if requested != 'auto':
            if not requested.startswith('cuda:'):
                raise ValueError('Expected auto or cuda:<visible index>')
            local = int(requested.split(':')[1])
            devices = [d for d in devices if d['visible_cuda_index'] == local]
        # Both active computation and resident allocations count toward load.
        # Admission still requires the measured free-memory check below.
        devices.sort(key=lambda d:(d['utilization']/100+d['used_mib']/d['total_mib'],
                                   len(d['processes']),-d['free_mib']))
        for device in devices:
            record = dict(device=device, retry=retry)
            try:
                measurement = probe(device)
                peak = float(measurement['peak_reserved_mib'])+float(measurement['context_mib'])
                if peak <= 0:
                    raise ValueError('Actual positive workload peak measurement required')
                fresh = next(d for d in discover() if d['uuid'] == device['uuid'])
                required = peak*(1+margin_fraction)
                record.update(measurement=measurement, required_mib=required, current_free_mib=fresh['free_mib'])
                if fresh['free_mib'] < required:
                    record['status'] = 'INSUFFICIENT_MEASURED_HEADROOM';attempts.append(record);continue
                record['status'] = 'ADMITTED';attempts.append(record)
                return dict(status='ADMITTED', selected=fresh, torch_device='cuda:0',
                            cuda_visible_devices=fresh['uuid'], attempts=attempts,
                            safety_margin_fraction=margin_fraction)
            except Exception as error:
                record.update(status='PROBE_FAILED',error=repr(error));attempts.append(record)
        if retry < retries:time.sleep(retry_seconds)
    error = WaitingForGPUResources('WAITING_FOR_GPU_RESOURCES')
    error.attempts = attempts
    raise error
