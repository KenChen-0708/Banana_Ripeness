param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)

$outputDirectory = $null
$inputPath = $null
for ($index = 0; $index -lt $Arguments.Count; $index++) {
    if ($Arguments[$index] -eq '--outdir' -and $index + 1 -lt $Arguments.Count) {
        $outputDirectory = $Arguments[$index + 1]
        $index++
    } elseif (Test-Path -LiteralPath $Arguments[$index] -PathType Leaf) {
        $inputPath = (Resolve-Path -LiteralPath $Arguments[$index]).Path
    }
}
if (-not $outputDirectory -or -not $inputPath) {
    Write-Error 'The Word conversion shim requires --outdir and an input document.'
    exit 2
}
$outputDirectory = [System.IO.Path]::GetFullPath($outputDirectory)
[System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
$pdfPath = Join-Path $outputDirectory (([System.IO.Path]::GetFileNameWithoutExtension($inputPath)) + '.pdf')
$word = $null
$document = $null
$powerPoint = $null
$presentation = $null
try {
    if ([System.IO.Path]::GetExtension($inputPath) -in @('.ppt', '.pptx')) {
        $powerPoint = New-Object -ComObject PowerPoint.Application
        $presentation = $powerPoint.Presentations.Open($inputPath, $true, $false, $false)
        $presentation.SaveAs($pdfPath, 32)
        Write-Output "convert $inputPath -> $pdfPath using Microsoft PowerPoint"
    } else {
        $word = New-Object -ComObject Word.Application
        $word.Visible = $false
        $word.DisplayAlerts = 0
        $document = $word.Documents.Open($inputPath, $false, $true)
        $document.ExportAsFixedFormat($pdfPath, 17)
        Write-Output "convert $inputPath -> $pdfPath using Microsoft Word"
    }
} finally {
    if ($presentation) { $presentation.Close() }
    if ($powerPoint) { $powerPoint.Quit() }
    if ($document) { $document.Close($false) }
    if ($word) { $word.Quit() }
}
