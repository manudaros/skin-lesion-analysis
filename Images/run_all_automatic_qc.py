from step02_validate_index_csv import (
    validate_index_csv,
)
from step03_check_image_readability import (
    run_image_readability_check,
)
from step04_check_dimensions import (
    run_dimension_check,
)
from step05_check_mask_quality import (
    run_mask_quality_checks,
)
from step06_summarize_dataset_statistics import (
    run_dataset_statistics,
)
from step07_check_duplicates import (
    run_duplicate_checks,
)
from step10_generate_qc_summary import (
    generate_qc_summary,
)


def main() -> None:
    """
    Run every automatic, non-interactive
    quality-control step.

    The one-time organization and index-creation
    scripts are not run automatically.

    The two overlay scripts are excluded because
    they open interactive Matplotlib windows.
    """
    checks = [
        (
            "Index validation",
            validate_index_csv,
        ),
        (
            "Original image readability",
            run_image_readability_check,
        ),
        (
            "Image-mask dimensions",
            run_dimension_check,
        ),
        (
            "Mask quality",
            run_mask_quality_checks,
        ),
        (
            "Dataset statistics",
            run_dataset_statistics,
        ),
        (
            "Duplicate files and IDs",
            run_duplicate_checks,
        ),
        (
            "Final QC summary",
            generate_qc_summary,
        ),
    ]

    failed_steps = []

    print("=" * 80)
    print(
        "STARTING AUTOMATIC DATASET "
        "QUALITY CONTROL"
    )
    print("=" * 80)

    for (
        step_number,
        (
            step_name,
            function,
        ),
    ) in enumerate(
        checks,
        start=1,
    ):
        print(
            "\n"
            + "=" * 80
        )

        print(
            f"STEP {step_number}: "
            f"{step_name}"
        )

        print("=" * 80)

        try:
            function()

        except Exception as error:
            failed_steps.append(
                step_name
            )

            print(
                f"Step failed: "
                f"{step_name}\n"
                f"{type(error).__name__}: "
                f"{error}"
            )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "AUTOMATIC QUALITY-CONTROL "
        "PIPELINE FINISHED"
    )

    print("=" * 80)

    if failed_steps:
        print(
            "The following steps failed:"
        )

        for step_name in failed_steps:
            print(
                f"- {step_name}"
            )

    else:
        print(
            "All automatic checks "
            "completed successfully."
        )

    print(
        "\nRun these two scripts "
        "manually afterwards:"
    )

    print(
        "- step08_check_img_mask_overlay.py"
    )

    print(
        "- step09_check_attribute_mask_overlay.py"
    )


if __name__ == "__main__":
    main()