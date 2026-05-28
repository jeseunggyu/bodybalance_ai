
$MY_ENDPOINT = "b7fa8783-3ffa-11f1-b85e-0ea3589134b3"
$N_SUBJECTS  = 150
$SPEEDS = @("W3", "W4")
# ─────────────────────────────────────

$FRDR_ENDPOINT = "f163c1b3-9c88-42f6-a7bb-5839ed6c4063"
$FRDR_BASE     = "/1/published/publication_1280/submitted_data"

# Windows 경로 → Globus 경로 변환
$LOCAL_PATH = "/C/bodybalance_ai/data/raw/public"

$batchFile = "$env:TEMP\globus_batch_w3w4.txt"
$lines = @()

foreach ($i in 1..$N_SUBJECTS) {
    $num = "{0:D3}" -f $i       # 001, 002, ...
    $id  = "P{0:D3}" -f $i     # P001, P002, ...

    foreach ($speed in $SPEEDS) {
        $lines += "${FRDR_BASE}/py/${num}/BF/${speed}/metadata.csv ${LOCAL_PATH}/${id}/${speed}_BF/metadata.csv"
        $lines += "${FRDR_BASE}/py/${num}/BF/${speed}/pipeline_1.npz ${LOCAL_PATH}/${id}/${speed}_BF/pipeline_1.npz"
    }
}

[System.IO.File]::WriteAllLines($batchFile, $lines, [System.Text.Encoding]::ASCII)

Write-Host "전송 목록: $($lines.Count)개 파일"

globus transfer $FRDR_ENDPOINT $MY_ENDPOINT `
    --batch $batchFile `
    --sync-level checksum `
    --label "BodyBalance_W3W4BF_150subjects"

Write-Host "완료"
