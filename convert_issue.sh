#!/bin/bash

# Script to convert a single markdown file to a GitHub Issue and add it to a GitHub Project
# Usage: ./convert_issue.sh <path_to_md_file>

set -e

INPUT_FILE="$1"

# 1. Validate input
if [ -z "$INPUT_FILE" ]; then
    echo "❌ Usage: $0 <path_to_md_file>"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ File not found: $INPUT_FILE"
    exit 1
fi

# 2. Check GitHub CLI
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI not installed (brew install gh)"
    exit 1
fi

# 3. Configuration (from create-all-issues.sh)
# ID проектов
PROJECT_BACKEND_ID="PVT_kwHOB2NmzM4BIlTq"
PROJECT_ML_ID="PVT_kwHOB2NmzM4BIlTd"
PROJECT_IOS_ID="PVT_kwHOB2NmzM4BIlT8"
PROJECT_GENERAL_ID="PVT_kwHOB2NmzM4BIlTL"

PROJECT_BACKEND_NUM=7
PROJECT_ML_NUM=6
PROJECT_IOS_NUM=8
PROJECT_GENERAL_NUM=5

# ID полей проектов (разные для каждого проекта!)
BACKEND_PRIORITY="PVTSSF_lAHOB2NmzM4BIlTqzg5ACvo"
BACKEND_SIZE="PVTSSF_lAHOB2NmzM4BIlTqzg5ACvs"
BACKEND_ESTIMATE="PVTF_lAHOB2NmzM4BIlTqzg5ACvw"

ML_PRIORITY="PVTSSF_lAHOB2NmzM4BIlTdzg5AClU"
ML_SIZE="PVTSSF_lAHOB2NmzM4BIlTdzg5AClY"
ML_ESTIMATE="PVTF_lAHOB2NmzM4BIlTdzg5AClc"

IOS_PRIORITY="PVTSSF_lAHOB2NmzM4BIlT8zg5AC7E"
IOS_SIZE="PVTSSF_lAHOB2NmzM4BIlT8zg5AC7I"
IOS_ESTIMATE="PVTF_lAHOB2NmzM4BIlT8zg5AC7M"

GENERAL_PRIORITY="PVTSSF_lAHOB2NmzM4BIlTLzg5ACZQ"
GENERAL_SIZE="PVTSSF_lAHOB2NmzM4BIlTLzg5ACZU"
GENERAL_ESTIMATE="PVTF_lAHOB2NmzM4BIlTLzg5ACZY"

# Опции Priority
PRIORITY_P0="79628723"
PRIORITY_P1="0a877460"
PRIORITY_P2="da944a9c"

# Опции Size
SIZE_XS="911790be"
SIZE_S="b277fb01"
SIZE_M="86db8eb3"
SIZE_L="853c8207"
SIZE_XL="2d0801e2"

# Helpers
get_component() {
    local file=$1
    local dir=$(dirname "$file")
    local component=$(basename "$dir")
    
    if [[ "$component" == "backend" || "$component" == "Backend" ]]; then
        echo "backend"
    elif [[ "$component" == "ml-service" || "$component" == "MLService" || "$component" == "ml-service" ]]; then
        echo "ml-service"
    elif [[ "$component" == "ios-app" || "$component" == "iOSApp" || "$component" == "ios-app" ]]; then
        echo "ios-app"
    elif [[ "$component" == "general" ]]; then
        echo "general"
    else
        # Fallback or error? For now, let's assume general if unknown
        echo "general"
    fi
}

get_priority() {
    local file=$1
    if grep -q "🔥 Critical\|Приоритет.*Critical" "$file"; then
        echo "P0"
    elif grep -q "⚡ High\|Приоритет.*High" "$file"; then
        echo "P1"
    else
        echo "P2"
    fi
}

get_size() {
    local file=$1
    if grep -q "час" "$file"; then
        echo "XS"
    elif grep -q "1-2.*дня\|2-3.*часа" "$file"; then
        echo "S"
    elif grep -q "3-5.*дня\|дня\|дней" "$file"; then
        echo "M"
    elif grep -q "1.*недел\|недел" "$file"; then
        echo "L"
    else
        echo "XL"
    fi
}

get_estimate() {
    local file=$1
    if grep -q "1-2.*часа" "$file"; then
        echo "2"
    elif grep -q "2-3.*часа" "$file"; then
        echo "3"
    elif grep -q "3-4.*часа" "$file"; then
        echo "4"
    elif grep -q "4-6.*часа" "$file"; then
        echo "5"
    elif grep -q "6-8.*часа" "$file"; then
        echo "7"
    elif grep -q "1-2.*дня\|1-2 дня" "$file"; then
        echo "12"
    elif grep -q "3-5.*дня\|3-5 дней" "$file"; then
        echo "32"
    elif grep -q "1.*недел" "$file"; then
        echo "40"
    elif grep -q "2.*недел" "$file"; then
        echo "80"
    elif grep -q "2-3.*недел" "$file"; then
        echo "100"
    else
        echo "40"
    fi
}

get_project_num() {
    case $1 in
        backend) echo "$PROJECT_BACKEND_NUM" ;;
        ml-service) echo "$PROJECT_ML_NUM" ;;
        ios-app) echo "$PROJECT_IOS_NUM" ;;
        general) echo "$PROJECT_GENERAL_NUM" ;;
    esac
}

get_project_id() {
    case $1 in
        backend) echo "$PROJECT_BACKEND_ID" ;;
        ml-service) echo "$PROJECT_ML_ID" ;;
        ios-app) echo "$PROJECT_IOS_ID" ;;
        general) echo "$PROJECT_GENERAL_ID" ;;
    esac
}

get_priority_field() {
    case $1 in
        backend) echo "$BACKEND_PRIORITY" ;;
        ml-service) echo "$ML_PRIORITY" ;;
        ios-app) echo "$IOS_PRIORITY" ;;
        general) echo "$GENERAL_PRIORITY" ;;
    esac
}

get_size_field() {
    case $1 in
        backend) echo "$BACKEND_SIZE" ;;
        ml-service) echo "$ML_SIZE" ;;
        ios-app) echo "$IOS_SIZE" ;;
        general) echo "$GENERAL_SIZE" ;;
    esac
}

get_estimate_field() {
    case $1 in
        backend) echo "$BACKEND_ESTIMATE" ;;
        ml-service) echo "$ML_ESTIMATE" ;;
        ios-app) echo "$IOS_ESTIMATE" ;;
        general) echo "$GENERAL_ESTIMATE" ;;
    esac
}

get_priority_id() {
    case $1 in
        P0) echo "$PRIORITY_P0" ;;
        P1) echo "$PRIORITY_P1" ;;
        P2) echo "$PRIORITY_P2" ;;
    esac
}

get_size_id() {
    case $1 in
        XS) echo "$SIZE_XS" ;;
        S) echo "$SIZE_S" ;;
        M) echo "$SIZE_M" ;;
        L) echo "$SIZE_L" ;;
        XL) echo "$SIZE_XL" ;;
    esac
}

clean_body() {
    local input_file=$1
    local output_file=$2
    
    # Remove metadata headers from the body
    awk '
    BEGIN { skip=0 }
    /^## Приоритет/ { skip=1; next }
    /^## Компонент/ { skip=1; next }
    /^## Зависимости/ { skip=1; next }
    /^## Оценка времени/ { skip=1; next }
    /^## / { skip=0 }
    !skip { print }
    ' "$input_file" > "$output_file"
}

# 4. Main Logic

COMPONENT=$(get_component "$INPUT_FILE")
# Title: First line of file, remove '# '
TITLE=$(head -n 1 "$INPUT_FILE" | sed 's/^# //')

PRIORITY=$(get_priority "$INPUT_FILE")
SIZE=$(get_size "$INPUT_FILE")
ESTIMATE=$(get_estimate "$INPUT_FILE")

PRIORITY_ID=$(get_priority_id "$PRIORITY")
SIZE_ID=$(get_size_id "$SIZE")
PROJECT_NUM=$(get_project_num "$COMPONENT")
PROJECT_ID=$(get_project_id "$COMPONENT")

FIELD_PRIORITY=$(get_priority_field "$COMPONENT")
FIELD_SIZE=$(get_size_field "$COMPONENT")
FIELD_ESTIMATE=$(get_estimate_field "$COMPONENT")

TEMP_BODY=$(mktemp)
trap "rm -f $TEMP_BODY" EXIT
clean_body "$INPUT_FILE" "$TEMP_BODY"

echo "🚀 Creating issue: $TITLE"
echo "   File: $INPUT_FILE"
echo "   Component: $COMPONENT"
echo "   Project Num: $PROJECT_NUM"
echo "   Priority: $PRIORITY | Size: $SIZE | Estimate: ${ESTIMATE}h"

# Create Issue (add label only for component)
ISSUE_URL=$(gh issue create \
    --title "$TITLE" \
    --body-file "$TEMP_BODY" \
    --label "$COMPONENT")

if [ $? -ne 0 ]; then
    echo "❌ Failed to create issue."
    exit 1
fi

echo "   ✅ Issue created: $ISSUE_URL"

# Add to Project and update fields
ISSUE_NUMBER=$(echo "$ISSUE_URL" | grep -oE '[0-9]+$')

if [ -n "$ISSUE_NUMBER" ] && [ -n "$PROJECT_NUM" ]; then
    echo "   ⚙️  Adding to Project #$PROJECT_NUM..."
    
    ITEM_ID=$(gh project item-add "$PROJECT_NUM" \
        --owner tzopiz \
        --url "$ISSUE_URL" \
        --format json 2>/dev/null | jq -r '.id' || echo "")
    
    if [ -n "$ITEM_ID" ]; then
        # Update Priority
        gh api graphql -f query="
        mutation {
          updateProjectV2ItemFieldValue(input: {
            projectId: \"$PROJECT_ID\"
            itemId: \"$ITEM_ID\"
            fieldId: \"$FIELD_PRIORITY\"
            value: {singleSelectOptionId: \"$PRIORITY_ID\"}
          }) {
            projectV2Item { id }
          }
        }" >/dev/null 2>&1
        
        # Update Size
        gh api graphql -f query="
        mutation {
          updateProjectV2ItemFieldValue(input: {
            projectId: \"$PROJECT_ID\"
            itemId: \"$ITEM_ID\"
            fieldId: \"$FIELD_SIZE\"
            value: {singleSelectOptionId: \"$SIZE_ID\"}
          }) {
            projectV2Item { id }
          }
        }" >/dev/null 2>&1
        
        # Update Estimate
        gh api graphql -f query="
        mutation {
          updateProjectV2ItemFieldValue(input: {
            projectId: \"$PROJECT_ID\"
            itemId: \"$ITEM_ID\"
            fieldId: \"$FIELD_ESTIMATE\"
            value: {number: $ESTIMATE}
          }) {
            projectV2Item { id }
          }
        }" >/dev/null 2>&1
        
        echo "   ✓ Project fields updated (Priority, Size, Estimate)"
    else
        echo "   ⚠️  Failed to add item to project (might verify permissions or project existence)"
    fi
fi

# 5. Branch Creation
echo ""
read -p "🌿 Would you like to start working on this issue now? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Determine branch prefix based on component
    case $COMPONENT in
        "backend") PREFIX="BMASTER" ;;
        "ios-app") PREFIX="IMASTER" ;;
        "ml-service") PREFIX="MLMASTER" ;;
        "general") PREFIX="GMASTER" ;;
        *) PREFIX="GMASTER" ;; # Default to General
    esac

    BRANCH_NAME="${PREFIX}-${ISSUE_NUMBER}"

    echo "🚀 Creating and checking out branch: $BRANCH_NAME"
    
    # Create and checkout branch associated with the issue
    # Using --checkout to switch to it
    gh issue develop "$ISSUE_NUMBER" --name "$BRANCH_NAME" --checkout
    
    if [ $? -eq 0 ]; then
        echo "✅ Switched to branch $BRANCH_NAME"
    else
        echo "❌ Failed to create branch."
    fi
fi

echo "🎉 Done!"
