#!/usr/bin/env python3
"""Compile real host-only production helpers against explicit CPU interfaces.

This is NOT an MLX, model, numerical, GPU or throughput qualification. Full SDK
and native tests remain separate. No package resolution or dependency download.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files = [
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/LayerCacheBankV2.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/MTP/MTPSampledTokenAssembly.swift',
        root/'Libraries/MLXLLM/Models/Gemma4DenseGateUpPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PrefillGluePolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PrefillNormalizationCarry.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4DeferredExpertState.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4DecodeGluePolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4RouterFinalistsPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4ScaledEmbeddingPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4QKVNormPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B8ExpertPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B8RoutePolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4PositionCycle.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4CacheRootPolicy.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B8RouteSources.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4B8RouteFoldSources.swift',
        root/'Libraries/MLXLMCommon/ContinuousBatchingV2/Gemma4RouterFinalistsSources.swift',
        root/'Tests/HostOnly/GemmaCatchup/CacheInterfaces.swift',
        root/'Tests/HostOnly/GemmaCatchup/RouteSortReference.swift',
        root/'Tests/HostOnly/GemmaCatchup/FinalistsReference.swift',
        root/'Tests/HostOnly/GemmaCatchup/HostContracts.swift',
    ]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    version = subprocess.run(['swiftc','--version'], capture_output=True, text=True, check=True).stdout.strip()
    runs = []
    with tempfile.TemporaryDirectory(prefix='gemma-host-contracts-') as directory:
        temp = Path(directory)
        for mode, flag in [('debug','-Onone'),('optimized','-O')]:
            binary = temp/f'contracts-{mode}'
            command = ['swiftc','-parse-as-library',flag,'-module-cache-path',str(temp/'modules'),
                       *map(str,files),'-o',str(binary)]
            build = subprocess.run(command, capture_output=True, text=True)
            if build.returncode:
                raise RuntimeError(build.stderr)
            test = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
            runs.append({'mode':mode,'build_exit':build.returncode,'exit':test.returncode,
                         'stdout':test.stdout.strip(),'stderr':test.stderr.strip()})
            print(json.dumps(runs[-1]), flush=True)
            if test.returncode:
                raise RuntimeError('CPU host contract assertion failed')
    receipt = {'schema_version':1,'scope':'real host helpers with CPU-only cache/tensor interface doubles',
               'swift':version,'sources_sha256':hashes,'runs':runs,'model_loaded':False,
               'mlx_imported':False,'gpu_work':False,'native_or_speed_qualified':False}
    if args.receipt:
        with args.receipt.open('x') as output:
            json.dump(receipt, output, indent=2)
            output.write('\n')


if __name__ == '__main__':
    main()
