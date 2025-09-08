#!/usr/bin/env python3
"""
Fix for LLaVA config conflict in transformers
Run this before importing LLaVA-Rad
"""

def fix_llava_config_conflict():
    """Remove conflicting llava config from transformers"""
    try:
        from transformers.models.auto import configuration_auto
        
        # Check if llava is in the extra content
        if hasattr(configuration_auto, 'CONFIG_MAPPING'):
            config_mapping = configuration_auto.CONFIG_MAPPING
            
            # Remove from _extra_content if it exists
            if hasattr(config_mapping, '_extra_content') and 'llava' in config_mapping._extra_content:
                del config_mapping._extra_content['llava']
                print("✓ Removed conflicting 'llava' from transformers config")
                return True
                
        # Also try the AutoConfig approach
        from transformers import AutoConfig
        if hasattr(AutoConfig, '_config_mapping') and hasattr(AutoConfig._config_mapping, '_extra_content'):
            if 'llava' in AutoConfig._config_mapping._extra_content:
                del AutoConfig._config_mapping._extra_content['llava']
                print("✓ Removed conflicting 'llava' from AutoConfig")
                return True
                
        print("No llava conflict found in transformers")
        return False
        
    except Exception as e:
        print(f"Error fixing config: {e}")
        return False


if __name__ == "__main__":
    fix_llava_config_conflict()