/**
 * PLUGIN INVENTORY — the reactive record of every desktop plugin the app
 * knows about (bundled `src/plugins/*`, the in-repo runtime example, the
 * `<hermes home>/desktop-plugins/*` disk door — incl. agent-written ones),
 * plus the persisted DISABLED set. The settings "Plugins" page renders this;
 * the loaders publish into it and consult the disabled set before
 * registering. Enable/disable is live: each record carries the loader's own
 * activate/deactivate handles, so toggling never needs an app reload.
 */

import { atom } from 'nanostores'

export type PluginKind = 'bundled' | 'disk' | 'runtime'
export type PluginStatus = 'disabled' | 'error' | 'loaded'

export interface PluginRecord {
  id: string
  name: string
  kind: PluginKind
  status: PluginStatus
  /** Load/registration failure message (status 'error'). */
  error?: string
  /** Absolute plugin.js path (disk plugins) — powers "Reveal in Finder". */
  file?: string
}

const DISABLED_KEY = 'hermes.desktop.disabledPlugins.v1'

function loadDisabled(): ReadonlySet<string> {
  try {
    const raw = window.localStorage.getItem(DISABLED_KEY)

    return new Set(raw ? (JSON.parse(raw) as string[]) : [])
  } catch {
    return new Set()
  }
}

export const $disabledPlugins = atom<ReadonlySet<string>>(loadDisabled())

export const pluginDisabled = (id: string) => $disabledPlugins.get().has(id)

function saveDisabled(next: ReadonlySet<string>) {
  $disabledPlugins.set(next)

  try {
    if (next.size === 0) {
      window.localStorage.removeItem(DISABLED_KEY)
    } else {
      window.localStorage.setItem(DISABLED_KEY, JSON.stringify([...next]))
    }
  } catch {
    // Nonfatal.
  }
}

export const $pluginRecords = atom<Record<string, PluginRecord>>({})

/** Loader-owned lifecycle handles, keyed by plugin id. */
const handles = new Map<string, { activate: () => Promise<void> | void; deactivate: () => void }>()

/** Publish/refresh a plugin's record + its activate/deactivate handles. */
export function publishPlugin(
  record: PluginRecord,
  handle?: { activate: () => Promise<void> | void; deactivate: () => void }
): void {
  $pluginRecords.set({ ...$pluginRecords.get(), [record.id]: record })

  if (handle) {
    handles.set(record.id, handle)
  }
}

export function patchPlugin(id: string, patch: Partial<PluginRecord>): void {
  const current = $pluginRecords.get()[id]

  if (current) {
    $pluginRecords.set({ ...$pluginRecords.get(), [id]: { ...current, ...patch } })
  }
}

export function dropPlugin(id: string): void {
  const { [id]: _dropped, ...rest } = $pluginRecords.get()
  $pluginRecords.set(rest)
  handles.delete(id)
}

/** Live toggle: deactivate + remember, or forget + reactivate. */
export async function setPluginEnabled(id: string, enabled: boolean): Promise<void> {
  const next = new Set($disabledPlugins.get())

  if (enabled) {
    next.delete(id)
  } else {
    next.add(id)
  }

  saveDisabled(next)

  const handle = handles.get(id)

  if (!handle) {
    return
  }

  if (enabled) {
    await handle.activate()
  } else {
    handle.deactivate()
    patchPlugin(id, { status: 'disabled' })
  }
}
