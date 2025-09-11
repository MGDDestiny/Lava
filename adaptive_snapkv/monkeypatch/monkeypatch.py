from importlib.metadata import version
import warnings
import transformers
import transformers.models.mistral.modeling_mistral
from adaptive_snapkv.monkeypatch.fixed_mistral_hijack import fixed_mistral_flash_attn2_forward, fixed_MistralModel_forward
from adaptive_snapkv.monkeypatch.fixed_mistral_hijack import prepare_inputs_for_generation_mistral as fixed_prepare_inputs_for_generation_mistral
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack import adaptive_mistral_flash_attn2_forward,adaptive_MistralModel_forward
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack import prepare_inputs_for_generation_mistral as ada_prepare_inputs_for_generation_mistral
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack_lava import adaptive_MistralModel_forward_lava, adaptive_mistral_flash_attn2_forward_lava
from adaptive_snapkv.monkeypatch.adaptive_mistral_hijack_lava import prepare_inputs_for_generation_mistral_lava as ada_prepare_inputs_for_generation_mistral_lava

from adaptive_snapkv.monkeypatch.fixed_qwen2_hijack import fixed_qwen2_flash_attn2_forward, fixed_Qwen2Model_forward
from adaptive_snapkv.monkeypatch.fixed_qwen2_hijack import prepare_inputs_for_generation_qwen2 as fixed_prepare_inputs_for_generation_qwen2
from adaptive_snapkv.monkeypatch.adaptive_qwen2_hijack import adaptive_qwen2_flash_attn2_forward, adaptive_Qwen2Model_forward
from adaptive_snapkv.monkeypatch.adaptive_qwen2_hijack import prepare_inputs_for_generation_qwen2 as ada_prepare_inputs_for_generation_qwen2
from adaptive_snapkv.monkeypatch.adaptive_qwen2_hijack_lava import adaptive_Qwen2Model_forward_lava, adaptive_qwen2_flash_attn2_forward_lava
from adaptive_snapkv.monkeypatch.adaptive_qwen2_hijack_lava import prepare_inputs_for_generation_qwen2_lava as ada_prepare_inputs_for_generation_qwen2_lava

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.41']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def replace_mistral_fixed():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = fixed_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = fixed_MistralModel_forward

def replace_mistral_adaptive():
    check_version()
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward

def replace_mistral_lava():
    transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_mistral_lava
    transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = adaptive_mistral_flash_attn2_forward_lava
    transformers.models.mistral.modeling_mistral.MistralModel.forward = adaptive_MistralModel_forward_lava

def replace_qwen2_fixed():
    check_version()
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = fixed_prepare_inputs_for_generation_qwen2
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = fixed_qwen2_flash_attn2_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = fixed_Qwen2Model_forward

def replace_qwen2_adaptive():
    check_version()
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_qwen2
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = adaptive_qwen2_flash_attn2_forward
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = adaptive_Qwen2Model_forward

def replace_qwen2_lava():
    check_version()
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = ada_prepare_inputs_for_generation_qwen2_lava
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = adaptive_qwen2_flash_attn2_forward_lava    
    transformers.models.qwen2.modeling_qwen2.Qwen2Model.forward = adaptive_Qwen2Model_forward_lava
