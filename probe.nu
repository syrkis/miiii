#!/usr/bin/env nu

# probe.nu - Flexible experiment querying for miiii
# Usage: nu probe.nu [command] [options]

const DB_PATH = "metrics.db"

# Base query helper
def query-db [sql: string] {
    open $DB_PATH | query db $sql
}

# Basic probes
def list-tasks [] {
    query-db "SELECT DISTINCT task FROM runs ORDER BY task"
    | table
}

def list-lrs [] {
    query-db "SELECT DISTINCT lr FROM runs ORDER BY lr"
    | table
}

# Command help
def print-help [] {
    echo "Available commands:"
    echo "  list-tasks              - List all unique tasks"
    echo "  list-lrs               - List all unique learning rates"
    echo ""
    echo "Examples:"
    echo "  nu probe.nu list-tasks"
}

# Main command dispatcher
def main [...args] {
    match ($args | length) {
        0 => { print-help }
        1 => {
            match $args.0 {
                "list-tasks" => { list-tasks }
                "list-lrs" => { list-lrs }
                # "top-performers" => { top-performers }
                _ => { print-help }
            }
        }
        _ => { print-help }
    }
}

# Run main if script is executed directly
if ($env.FILE_PWD | path basename) == "probe.nu" {
    main ($env.ARGS | range 1..)
}
