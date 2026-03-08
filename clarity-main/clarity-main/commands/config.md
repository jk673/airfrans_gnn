---
model: claude-opus-4-6
description: Set where Clarity saves review files.
argument-hint: [save-path]
---

# /clarity:config

Manage Clarity plugin configuration.

## Behaviour

### If an argument is provided

The argument is a file path. Set it as the save location for all Clarity review files.

1. Validate the path exists and is a directory. If it does not exist, ask the user if they want to create it. If they decline, or if directory creation fails, display an error and do not save the configuration. Suggest running `/clarity:config` again with a valid path.
2. Save the configuration to `${CLAUDE_PLUGIN_ROOT}/.config.json` as:
   ```json
   {
     "saveLocation": "[absolute path]"
   }
   ```
3. Confirm:
   > Save location set to: [path]
   > Clarity reviews will be saved here.

### If no argument is provided

Display the current configuration:

1. Read `${CLAUDE_PLUGIN_ROOT}/.config.json` if it exists
2. Display:
   > **Clarity Configuration**
   > - Save location: [configured path, or "Not configured (using clarity/ in current working directory)"]
   >
   > To change the save location, run `/clarity:config [path]`

### Configuration file

Store configuration in `${CLAUDE_PLUGIN_ROOT}/.config.json`. If this file does not exist, all commands default to creating a `clarity/` directory in the current working directory.
